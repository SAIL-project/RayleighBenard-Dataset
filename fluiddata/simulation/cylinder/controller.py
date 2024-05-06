class ControllerPlugin(BasePlugin):
    name = "controller"
    systems = ["*"]
    formulations = ["dual", "std"]

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Process frequency and other params
        self.nsteps = self.cfg.getint(cfgsect, "nsteps")
        self.save_data = self.cfg.getint(cfgsect, "savedata")
        self.set_omega = self.cfg.getint(cfgsect, "setomega")
        self.perform_mpc = self.cfg.getint(cfgsect, "mpc") == 1

        # List of points to be sampled and format
        self.pts = self.cfg.getliteral(cfgsect, "samp-pts")
        self.fmt = self.cfg.get(cfgsect, "format", "primitive")

        # Define directory where solution snapshots should be saved
        self.save_dir = self.cfg.getpath(cfgsect, "save_dir")

        # If performing mpc, then load network
        if self.perform_mpc:

            # Define checkpoint name
            self.ckpt_name = self.cfg.get(cfgsect, "checkpoint")

            # Define directory containing training scripts
            self.training_path = self.cfg.getpath(cfgsect, "training_path")
            sys.path.append(self.training_path)

            # Define directory containing base flow solution
            base_flow = self.cfg.getpath(cfgsect, "base_flow")

            # Set constraints for mpc
            self.R = self.cfg.getfloat(cfgsect, "R")
            self.u_max = self.cfg.getfloat(cfgsect, "u_max")

            # Read in args
            with open(self.training_path + "/args.json") as args_dict:
                args_dict = json.load(
                    args_dict,
                )
            self.args = argparse.Namespace()
            for k, v in args_dict.items():
                vars(self.args)[k] = v

            # Define array to hold old time snapshots and control inputs of the system
            self.X = np.zeros((self.args.seq_length // 2 + 1, 128, 256, 4), dtype=np.float32)
            self.u = np.zeros((self.args.seq_length // 2, self.args.action_dim), dtype=np.float32)

            # Initialize predicted state
            self.x_pred = np.zeros(self.args.code_dim)

            # Run script to find desired attributes and store them
            command = (
                "python "
                + self.training_path
                + "/find_matrices.py "
                + self.training_path
                + " "
                + self.ckpt_name
                + " "
                + base_flow
            )
            subprocess.call(command.split())

            # Load desired attributes from file
            f = h5py.File("./matrices_misc.h5", "r")
            self.B = np.array(f["B"])
            self.goal_state = np.array(f["goal_state"])

        # Initial omega
        intg.system.omega = 0

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # MPI rank responsible for each point and rank-indexed info
        self._ptsrank = ptsrank = []
        self._ptsinfo = ptsinfo = [[] for i in range(comm.size)]

        # Physical location of the solution points
        plocs = [p.swapaxes(1, 2) for p in intg.system.ele_ploc_upts]

        # Load map from point to index
        with open("loc_to_idx.json") as loc_to_idx:
            loc_to_idx_str = json.load(
                loc_to_idx,
            )
            self.loc_to_idx = dict()
            for key in loc_to_idx_str:
                self.loc_to_idx[int(key)] = loc_to_idx_str[key]

        # Locate the closest solution points in our partition
        closest = _closest_upts(intg.system.ele_types, plocs, self.pts)

        # Process these points
        for cp in closest:
            # Reduce over the distance
            _, mrank = comm.allreduce((cp[0], rank), op=get_mpi("minloc"))

            # Store the rank responsible along with its info
            ptsrank.append(mrank)
            ptsinfo[mrank].append(comm.bcast(cp[1:] if rank == mrank else None, root=mrank))

    def _process_samples(self, samps):
        samps = np.array(samps)

        # If necessary then convert to primitive form
        if self.fmt == "primitive" and samps.size:
            samps = self.elementscls.con_to_pri(samps.T, self.cfg)
            samps = np.array(samps).T

        return samps.tolist()

    # Find A-matrix and initial code value from neural network
    def _find_dynamics(self):
        # Save X and u to file
        f = h5py.File("./X_u.h5", "w")
        f["X"] = self.X
        f["u"] = self.u
        f.close()

        # Run python script to find A matrix and initial state
        command = (
            "python "
            + self.training_path
            + "/find_dynamics.py "
            + self.training_path
            + " "
            + self.ckpt_name
        )
        subprocess.call(command.split())

        # Load desired values from file and return
        f = h5py.File("A_x0.h5", "r")
        A = np.array(f["A"])
        x0 = np.array(f["x0"])

        return A, x0

    # Perform MPC optimization to find next input
    # Following example from CVXPY documentation
    def _find_mpc_input(self, A, B, x0):
        # First define prediction horizon
        T = 16

        # Define variables
        x = Variable(shape=(self.args.code_dim, T + 1))
        u = Variable(shape=(self.args.action_dim, T))

        # Define costs for states and inputs
        Q = np.eye(self.args.code_dim)
        R = self.R * np.eye(self.args.action_dim)

        # Construct and solve optimization problem
        cost = 0
        constr = []
        for t in range(T):
            cost += quad_form((x[:, t + 1] - self.goal_state), Q) + quad_form(u[:, t], R)
            constr += [
                x[:, t + 1] == A * x[:, t] + (B * u[:, t])[:, 0],
                norm(u[:, t], "inf") <= self.u_max,
            ]

        # Sum problem objectives and concatenate constraints
        constr += [x[:, 0] == x0]
        prob = Problem(Minimize(cost), constr)
        prob.solve()

        x1 = np.array([x.value[i, 1] for i in range(x.value.shape[0])])
        try:
            return u.value[0, 0]  # Change if not scalar input
        except:
            return 0.0

    def __call__(self, intg):
        # Return if there is nothing to do for this step
        if intg.nacptsteps % self.nsteps:
            return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))

        # Points we're responsible for sampling
        ourpts = self._ptsinfo[comm.rank]

        # Sample the solution matrices at these points
        samples = [solns[et][ui, :, ei] for _, et, (ui, ei) in ourpts]
        samples = self._process_samples(samples)

        # Gather to the root rank to give a list of points per rank
        samples = comm.gather(samples, root=root)

        # If we're the root rank process the data
        if rank == root:
            data = []

            # Collate
            iters = [zip(pi, sp) for pi, sp in zip(self._ptsinfo, samples)]

            for mrank in self._ptsrank:
                # Unpack
                (ploc, etype, idx), samp = next(iters[mrank])

                # Determine the physical mesh rank
                prank = intg.rallocs.mprankmap[mrank]

                # Prepare the output row [[x, y], [rho, rhou, rhouv, E]]
                row = [ploc, samp]

                # Append
                data.append(row)

            # Define info for saving to file
            list_of_files = glob.glob(self.save_dir + "/*")
            if len(list_of_files) == 0:
                file_num = 0
            else:
                latest_file = max(list_of_files, key=os.path.getctime)
                file_num = int(latest_file[-7:-3])

            # Save data in desired format
            # Define freestream values for to be used for cylinder
            rho = 1.0
            P = 1.0
            u = 0.236
            v = 0.0
            e = P / rho / 0.4 + 0.5 * (u**2 + v**2)
            freestream = np.array([rho, rho * u, rho * v, e])
            sol_data = np.zeros((128, 256, 4))
            sol_data[:, :] = freestream
            for i in range(len(self.loc_to_idx)):
                idx1, idx2 = self.loc_to_idx[i]
                sol_data[idx1, idx2] = data[i][1]

            # Update running total of previous states
            if self.perform_mpc:
                self.X = np.vstack((self.X[1:], np.expand_dims(sol_data, axis=0)))

            # Initialize values
            t = intg.tcurr
            self.t_old = t
            pred_error = 0.0

            if self.set_omega == 0:
                omega = 0.0
            elif self.perform_mpc:
                # Find model of system and determine optimal input with MPC
                try:
                    A, x0 = self._find_dynamics()
                    if np.linalg.norm(self.X[0]) > 0.0:
                        u0 = self.u[-1]
                        omega = self._find_mpc_input(A, self.B, x0)
                    else:
                        omega = 0.0  # No input if insufficient data to construct dynamical model
                except:
                    print("Had an error in mpc -- setting omega to 0")
                    omega = 0.0
                self.u = np.concatenate((self.u[1:], np.expand_dims(np.array([omega]), axis=0)))
            else:
                # To generate training data
                # Have no inputs for periods, otherwise sinusoidal
                if t % 1000 > 900:
                    omega = 0.0
                else:
                    freq = 2 * np.pi * (t / 1000) / 500.0
                    omega = 0.3 * np.sin(freq * t)

                # Proportional control
                # location = 88 # re50
                # gain = 0.4 # re50
                # rho = sol_data[64, location, 0]
                # rho_v = sol_data[64, location, 2]
                # omega = gain*rho_v/rho

            # Save data if desired
            if self.save_data == 1:
                # Save to h5 file
                file_num += 1
                filename = self.save_dir + "/sol_data_" + str(file_num).zfill(4) + ".h5"
                f = h5py.File(filename, "w")
                f["sol_data"] = sol_data
                f["control_input"] = omega
                if self.perform_mpc:
                    f["cost"] = np.linalg.norm(self.goal_state - x0)
                f.close()
        else:
            omega = None

        # Broadcast omega to all of the MPI ranks
        intg.system.omega = float(comm.bcast(omega, root=root))
