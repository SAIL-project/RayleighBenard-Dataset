import copy
import warnings
from pathlib import Path

import numpy as np
import sympy
from mpi4py import MPI
from shenfun import (
    Array,
    Dx,
    Function,
    FunctionSpace,
    ShenfunFile,
    TensorProductSpace,
    TestFunction,
    chebyshev,
    div,
    grad,
    la,
)

from .channelflow2d import KMM

# global settings for numpy, sympy and MPI
x, y, tt = sympy.symbols("x,y,t", real=True)
comm = MPI.COMM_SELF
warnings.filterwarnings("ignore")


class RayleighBenard(KMM):
    def __init__(
        self,
        N_state=(64, 96),
        N_obs=(8, 32),
        domain=((-1, 1), (0, 2 * sympy.pi)),
        Ra=10000.0,
        Pr=0.7,
        dt=0.025,
        bcT=(2, 1),
        filename="data/shenfun/RB_2D",
        padding_factor=(1, 1.5),
        modsave=10000,
        checkpoint=10,
        family="C",
    ):
        """Form a complex number.

        Keyword arguments:

        N -- gridsize of collocation points (default: (64, 96))

        TODO
        """
        KMM.__init__(
            self,
            N=N_state,
            domain=domain,
            nu=np.sqrt(Pr / Ra),
            dt=dt,
            conv=0,
            filename=filename,
            family=family,
            padding_factor=padding_factor,
            modsave=modsave,
            checkpoint=checkpoint,
            dpdy=0,
        )
        # parameters
        self.kappa = 1.0 / np.sqrt(Pr * Ra)  # thermal diffusivity
        self.bcT = bcT
        self.bcT_avg = bcT  # datamember to remember the desired average temps
        self.domain = domain
        dt = self.dt
        kappa = self.kappa

        # Additional spaces and functions for Temperature equation
        self.T0 = FunctionSpace(N_state[0], family, bc=bcT, domain=domain[0])
        self.TT = TensorProductSpace(
            comm, (self.T0, self.F1), modify_spaces_inplace=True
        )  # Temperature
        self.uT_ = Function(self.BD)  # Velocity vector times T
        self.T_ = Function(self.TT)  # Temperature solution
        self.Tb = Array(self.TT)

        # Create files
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self.file_T = ShenfunFile(
            "_".join((filename, "T")), self.TT, backend="hdf5", mode="w", mesh="uniform"
        )

        # Modify checkpoint file
        self.checkpoint.data["0"]["T"] = [self.T_]

        # Chebyshev matrices are not sparse, so need a tailored solver. Legendre has simply 5
        # nonzero diagonals
        sol2 = chebyshev.la.Helmholtz if self.B0.family() == "chebyshev" else la.SolverGeneric1ND

        # Addition to u equation.
        self.pdes["u"].N = [self.pdes["u"].N, Dx(self.T_, 1, 2)]
        self.pdes["u"].latex += r"\frac{\partial^2 T}{\partial y^2}"

        # Remove constant pressure gradient from v0 equation
        self.pdes1d["v0"].N = self.pdes1d["v0"].N[0]

        # Add T equation
        q = TestFunction(self.TT)
        self.pdes["T"] = self.PDE(
            q,
            self.T_,
            lambda f: kappa * div(grad(f)),
            -div(self.uT_),
            dt=self.dt,
            solver=sol2,
            latex=r"\frac{\partial T}{\partial t} = \kappa \nabla^2 T - \nabla \cdot \vec{u}T",
        )

        # Observation outputs
        self.state = None
        self.obs = None
        self.obs_flat = None

        # Others
        self.obsGrid = N_obs
        self.Nstep = (N_state[0] // self.obsGrid[0], N_state[1] // self.obsGrid[1])

    def update_bc(self, t):
        # Update time-dependent bcs.
        self.T0.bc.update(t)
        self.T_.get_dealiased_space(self.padding_factor).bases[0].bc.update(t)

    def prepare_step(self, rk):
        self.convection()
        Tp = self.T_.backward(padding_factor=self.padding_factor)
        self.uT_ = self.up.function_space().forward(self.up * Tp, self.uT_)

    def tofile(self, tstep):
        self.file_u.write(tstep, {"u": [self.u_.backward(mesh="uniform")]}, as_scalar=True)
        self.file_T.write(tstep, {"T": [self.T_.backward(mesh="uniform")]})

    def init_from_checkpoint(self, filename=None):
        old_filename = self.checkpoint.filename
        if filename is not None:
            self.checkpoint.filename = (
                filename  # temporarily switch the filename of the Checkpoint instance
            )
        self.checkpoint.read(self.u_, "U", step=0)
        self.checkpoint.read(self.T_, "T", step=0)
        self.checkpoint.open()
        tstep = self.checkpoint.f.attrs["tstep"]
        t = self.checkpoint.f.attrs["t"]
        self.checkpoint.close()
        # restore the old filename of the Checkpoint
        # instance (which was changed if filename is given to function)
        self.checkpoint.filename = old_filename
        self.checkpoint.f = None
        return t, tstep

    # TODO: MS: look more into the initialization
    def initialize(self, rand=0.001, filename=None, np_random=None):
        if filename is not None:
            t, tstep = self.init_from_checkpoint(filename)
            self.update_bc(t)
            return t, tstep

        if np_random is None:
            np_random = np.random.default_rng()

        X = self.X

        if self.bcT[0] == 1:
            funT = 1
        elif int(self.bcT[0]) == 0.6:
            funT = 3
        elif int(self.bcT[0]) == 2:
            funT = 4
        else:
            funT = 2
        fun = {1: 1, 2: (0.9 + 0.1 * np.sin(2 * X[1])), 3: 0.6, 4: 2.0}[funT]
        self.Tb[:] = 0.5 * (
            1
            + 0.5 * self.bcT[1]
            - X[0] / (1 + self.bcT[1])
            + 0.125 * (2 - self.bcT[1]) * np.sin(np.pi * X[0])
        ) * fun + rand * np_random.standard_normal(self.Tb.shape) * (1 - X[0]) * (1 + X[0])
        self.T_ = self.Tb.forward(self.T_)
        self.T_.mask_nyquist(self.mask)
        return 0, 0

    def outputs(self):
        ub = self.u_.backward(self.ub)
        Tb = self.T_.backward(self.Tb)

        # state
        state = np.zeros((3, self.N[0], self.N[1]))
        state[0:2] = ub
        state[2] = Tb
        self.state = state

        # obs
        obs = np.zeros((3, self.obsGrid[0], self.obsGrid[1]))
        obs[0] = ub[1, :: self.Nstep[0], :: self.Nstep[1]]  # horizontal (x) axis
        obs[1] = ub[0, :: self.Nstep[0], :: self.Nstep[1]]  # vertical (y) axis
        obs[2] = Tb[:: self.Nstep[0], :: self.Nstep[1]]
        self.obs = obs

        # obs_flat
        obs_flat = copy.copy(obs)
        # obs_flat[0] *= 1.5    # M: I have no idea what this up-scaling is for, so I commented it out
        # obs_flat[1] *= 1.5
        # M: The following linear transformation is mapping the the temperature to [0.4, 4], no idea why this is done currently
        # So I commented it out, since it also interferes with the limits of the Box of the observation space in the environment.
        # obs_flat[2] = 2 * (obs_flat[2] - 0.8) 
        obs_flat = obs_flat.reshape(
            3 * self.obsGrid[0] * self.obsGrid[1],
        )
        self.obs_flat = obs_flat

    def compute_nusselt(self, from_obs=True):
        """
        Computes the nusselt number.
        from_obs: if True, computes the Nusselt number on the sparse observation,
            otherwise on the full state
        """
        div = (
            self.kappa
            * (self.bcT_avg[0] - self.bcT_avg[1])
            / (self.domain[0][1] - self.domain[0][0])
        )  # H = 2, Tb = 2.

        if from_obs:
            uyT_ = np.mean(np.mean(np.multiply(self.obs[1], self.obs[2]), axis=1), axis=0)
            T_ = np.mean(np.gradient(np.mean(self.obs[2], axis=1), axis=0))
        else:
            # MS: CAUTION HERE, IN THE STATE Y VELOCITIES ARE IN THE FIRST INDEX
            uyT_ = np.mean(np.mean(np.multiply(self.state[0], self.state[2]), axis=1), axis=0)
            T_ = np.mean(np.gradient(np.mean(self.state[2], axis=1), axis=0))

        return (uyT_ - self.kappa * T_) / div

    def compute_kinematic_energy(self):
        u2_xy = self.obs[1] * self.obs[1] + self.obs[0] * self.obs[0]
        return np.sum(u2_xy)

    def update_actuation(self, new_bcT):
        self.bcT = new_bcT
        self.T0.bc.bc["left"]["D"] = self.bcT[0]
        self.T0.bc.update()
        self.T0.bc.set_tensor_bcs(self.T0, self.T0.tensorproductspace)
        TP0 = self.T_.get_dealiased_space(self.padding_factor).bases[0]
        TP0.bc.bc["left"]["D"] = self.bcT[0]
        TP0.bc.update()
        TP0.bc.set_tensor_bcs(TP0, TP0.tensorproductspace)

    def step(self, t=0, tstep=0):
        # TODO what is c
        c = self.pdes["u"].stages()[2]

        # PDE solver steps
        for rk in range(self.PDE.steps()):
            self.prepare_step(rk)
            for eq in ["u", "T"]:
                self.pdes[eq].compute_rhs(rk)
            for eq in ["u"]:
                self.pdes[eq].solve_step(rk)
            self.compute_v(rk)
            self.update_bc(t + self.dt * c[rk + 1])
            self.pdes["T"].solve_step(rk)

        # update checkpoint and data files
        self.checkpoint.update(t, tstep)
        if tstep % self.modsave == 0:
            self.tofile(tstep)

        # update outputs and time
        self.outputs()
        return t + self.dt, tstep + 1

    def clean(self):
        self.TT.destroy()
        self.TB.destroy()
        self.TD.destroy()
        self.TC.destroy()
        self.TDp.destroy()
        self.BD.destroy()
        self.CD.destroy()