import warnings
from copy import copy
from pathlib import Path

import numpy as np
import sympy
from mpi4py import MPI
from shenfun import (
    Array,
    Checkpoint,
    Dx,
    Function,
    FunctionSpace,
    TensorProductSpace,
    TestFunction,
    chebyshev,
    div,
    grad,
    la,
)

from rbcdata.utils.sympy_helper import evalf_tuple

from .channelflow2d import KMM

# global settings for numpy, sympy and MPI
x, y, tt = sympy.symbols("x,y,t", real=True)
comm = MPI.COMM_SELF
warnings.filterwarnings("ignore")


class RayleighBenard(KMM):
    def __init__(
        self,
        N_state=(64, 96),
        N_obs=(8, 48),
        domain=((-1, 1), (0, 2 * sympy.pi)),
        Ra=10000.0,
        Pr=0.7,
        dt=0.025,
        bcT=(2, 1),
        checkpoint=None,
        padding_factor=(1, 1.5),
        modsave=10000,
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
            family=family,
            padding_factor=padding_factor,
            modsave=modsave,
            dpdy=0,
        )
        # parameters
        self.kappa = 1.0 / np.sqrt(Pr * Ra)  # thermal diffusivity
        self.bcT = bcT
        self.bcT_avg = copy(bcT)  # datamember to remember the desired average temps
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

        # checkpoint
        self.checkpoint = None
        if checkpoint is not None:
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint = Checkpoint(
                checkpoint,
                checkevery=10,
                data={"0": {"U": [self.u_], "T": [self.T_]}},
            )

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

        # Construct probes grid
        self.state = None
        self.obs = None

        # get observation grid
        self.N_obs = N_obs
        domain = evalf_tuple(domain)
        spaces = []
        for d, N in zip(domain, N_obs):
            step = (d[1] - d[0]) / N
            spaces.append(np.linspace(d[0] + (step / 2), d[1] - (step / 2), N))
        s1, s2 = np.meshgrid(spaces[1], spaces[0])
        s2 = np.flipud(s2)  # TODO TM: Why is it flipped?
        self.obs_points = np.vstack([s2.ravel(), s1.ravel()])

    def update_bc(self, t):
        # Update time-dependent bcs.
        self.T0.bc.update(t)
        self.T_.get_dealiased_space(self.padding_factor).bases[0].bc.update(t)

    def prepare_step(self, rk):
        self.convection()
        Tp = self.T_.backward(padding_factor=self.padding_factor)
        self.uT_ = self.up.function_space().forward(self.up * Tp, self.uT_)

    def init_from_checkpoint(self, filename):
        checkpoint = Checkpoint(
            filename,
            checkevery=10,
            data={"0": {"U": [self.u_], "T": [self.T_]}},
        )

        checkpoint.read(self.u_, "U", step=0)
        checkpoint.read(self.T_, "T", step=0)
        checkpoint.open()
        tstep = checkpoint.f.attrs["tstep"]
        t = self.checkpoint.f.attrs["t"]
        checkpoint.close()

        return t, tstep

    # TODO: MS: look more into the initialization
    def initialize(self, rand=0.001, checkpoint=None, np_random=None):
        if checkpoint is not None:
            t, tstep = self.init_from_checkpoint(checkpoint)
            self.update_bc(t)
            return t, tstep

        if np_random is None:
            np_random = np.random.default_rng()

        X = self.X

        if self.bcT_avg[0] == 1:
            funT = 1
        elif int(self.bcT_avg[0]) == 0.6:  # TODO TM: how can this be ever true?
            funT = 3
        elif int(self.bcT_avg[0]) == 2:
            funT = 4
        else:
            funT = 2
        fun = {1: 1, 2: (0.9 + 0.1 * np.sin(2 * X[1])), 3: 0.6, 4: 2.0}[funT]
        self.Tb[:] = 0.5 * (
            1
            + 0.5 * self.bcT_avg[1]
            - X[0] / (1 + self.bcT_avg[1])
            + 0.125 * (2 - self.bcT_avg[1]) * np.sin(np.pi * X[0])
        ) * fun + rand * np_random.standard_normal(self.Tb.shape) * (1 - X[0]) * (1 + X[0])
        self.T_ = self.Tb.forward(self.T_)
        self.T_.mask_nyquist(self.mask)
        return 0, 0

    def get_state(self):
        return self.state

    def get_obs(self):
        return self.obs

    def compute_outputs(self):
        # evaluate functions at collocation points
        ub = self.u_.backward(self.ub)
        Tb = self.T_.backward(self.Tb)

        # construct state
        state = np.zeros((3, self.N[0], self.N[1]))
        state[0:2] = ub
        state[2] = Tb

        # construct observation
        h, w = self.N_obs
        obs = np.zeros((3, h, w))
        obs[0:2] = self.u_.eval(self.obs_points).reshape(-1, h, w)
        obs[2] = self.T_.eval(self.obs_points).reshape(-1, h, w)

        self.state = state
        self.obs = obs

    def compute_nusselt(self, state):
        div = (
            self.kappa
            * (self.bcT_avg[0] - self.bcT_avg[1])
            / (self.domain[0][1] - self.domain[0][0])
        )  # H = 2, Tb = 2.

        uyT_ = np.mean(np.mean(np.multiply(state[0], state[2]), axis=1), axis=0)
        T_ = np.mean(np.gradient(np.mean(state[2], axis=1), axis=0))

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

        # update checkpoint
        if self.checkpoint is not None:
            self.checkpoint.update(t, tstep)

        # update outputs and time
        self.compute_outputs()
        return t + self.dt, tstep + 1

    def clean(self):
        self.TT.destroy()
        self.TB.destroy()
        self.TD.destroy()
        self.TC.destroy()
        self.TDp.destroy()
        self.BD.destroy()
        self.CD.destroy()
