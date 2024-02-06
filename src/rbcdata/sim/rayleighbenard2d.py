import copy
import warnings
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import sympy

from rbcdata.utils.rbc_simulation_params import RBCSimulationParams

try:
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
except ImportError:
    print("please install shenfun package to generate RBC data")

from .channelflow2d import KMM

# global settings for numpy, sympy and MPI
x, y, tt = sympy.symbols("x,y,t", real=True)
comm = MPI.COMM_SELF
warnings.filterwarnings("ignore")


class RayleighBenard2D(KMM):
    def __init__(self, params: RBCSimulationParams) -> None:
        """Form a complex number.

        Keyword arguments:

        N -- gridsize of collocation points (default: (64, 96))

        TODO
        """
        KMM.__init__(self, params=params)

        # parameters
        kappa = 1.0 / np.sqrt(params.Pr * params.Ra)
        self.kappa = kappa
        self.bcT = tuple(params.bcT)

        # Additional spaces and functions for Temperature equation
        self.T0 = FunctionSpace(
            params.N[0], params.family, bc=self.bcT, domain=self.domain[0]
        )
        self.TT = TensorProductSpace(
            comm, (self.T0, self.F1), modify_spaces_inplace=True
        )  # Temperature
        self.uT_ = Function(self.BD)  # Velocity vector times T
        self.T_ = Function(self.TT)  # Temperature solution
        self.Tb = Array(self.TT)

        # Create files
        Path(params.filename).parent.mkdir(parents=True, exist_ok=True)
        self.file_T = ShenfunFile(
            "_".join((params.filename, "T")),
            self.TT,
            backend="hdf5",
            mode="w",
            mesh="uniform",
        )

        # Modify checkpoint file
        self.checkpoint.data["0"]["T"] = [self.T_]

        # Chebyshev matrices are not sparse, so need a tailored solver. Legendre has simply 5 nonzero diagonals
        sol2 = (
            chebyshev.la.Helmholtz
            if self.B0.family() == "chebyshev"
            else la.SolverGeneric1ND
        )

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
            dt=params.dt,
            solver=sol2,
            latex=r"\frac{\partial T}{\partial t} = \kappa \nabla^2 T - \nabla \cdot \vec{u}T",
        )

        # Others
        self.obsGrid = params.Nobs
        self.Nstep = (params.N[0] // self.obsGrid[0], params.N[1] // self.obsGrid[1])

    def update_bc(self, t: float) -> None:
        # Update time-dependent bcs.
        self.T0.bc.update(t)
        self.T_.get_dealiased_space(self.padding).bases[0].bc.update(t)

    def prepare_step(self, rk: Any) -> None:
        self.convection()
        Tp = self.T_.backward(padding_factor=self.padding)
        self.uT_ = self.up.function_space().forward(self.up * Tp, self.uT_)

    def tofile(self, tstep: int) -> None:
        self.file_u.write(
            tstep, {"u": [self.u_.backward(mesh="uniform")]}, as_scalar=True
        )
        self.file_T.write(tstep, {"T": [self.T_.backward(mesh="uniform")]})

    def init_from_checkpoint(self) -> Tuple[float, int]:
        self.checkpoint.read(self.u_, "U", step=0)
        self.checkpoint.read(self.T_, "T", step=0)
        self.checkpoint.open()
        tstep = self.checkpoint.f.attrs["tstep"]
        t = self.checkpoint.f.attrs["t"]
        self.checkpoint.close()
        return t, tstep

    def initialize(self, from_checkpoint: bool = False) -> Tuple[float, int]:
        if from_checkpoint:
            self.checkpoint.read(self.u_, "U", step=0)
            self.checkpoint.read(self.T_, "T", step=0)
            self.checkpoint.open()
            tstep = self.checkpoint.f.attrs["tstep"]
            t = self.checkpoint.f.attrs["t"]
            self.checkpoint.close()
            self.update_bc(t)
            return t, tstep

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
        ) * fun + 0.001 * np.random.randn(*self.Tb.shape) * (1 - X[0]) * (1 + X[0])
        self.T_ = self.Tb.forward(self.T_)
        self.T_.mask_nyquist(self.mask)
        return 0, 0

    def outputs(self) -> None:
        ub = self.u_.backward(self.ub)
        Tb = self.T_.backward(self.Tb)

        # state
        state = np.zeros((3, self.params.N[0], self.params.N[1]))
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
        obs_flat[0] *= 1.5
        obs_flat[1] *= 1.5
        obs_flat[2] = 2 * (obs_flat[2] - 0.8)
        obs_flat = obs_flat.reshape(
            3 * self.obsGrid[0] * self.obsGrid[1],
        )
        self.obs_flat = obs_flat

    def compute_nusselt(self) -> float:
        div = self.kappa * (2.0 - self.bcT[1]) / 2  # H = 2, Tb = 2.

        uyT_ = np.mean(np.mean(np.multiply(self.obs[1], self.obs[2]), axis=1), axis=0)
        T_ = np.mean(np.gradient(np.mean(self.obs[2], axis=1), axis=0))
        return float((uyT_ - self.kappa * T_) / div)

    def compute_kinematic_energy(self) -> float:
        u2_xy = self.obs[1] * self.obs[1] + self.obs[0] * self.obs[0]
        return float(np.sum(u2_xy))

    def update_actuation(self, new_bcT: Tuple[float, float]) -> None:
        self.bcT = new_bcT
        self.T0.bc.bc["left"]["D"] = self.bcT[0]
        self.T0.bc.update()
        self.T0.bc.set_tensor_bcs(self.T0, self.T0.tensorproductspace)
        TP0 = self.T_.get_dealiased_space(self.padding).bases[0]
        TP0.bc.bc["left"]["D"] = self.bcT[0]
        TP0.bc.update()
        TP0.bc.set_tensor_bcs(TP0, TP0.tensorproductspace)

    def step(self, t: float = 0, tstep: int = 0) -> Tuple[float, int]:
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
            self.update_bc(t + self.params.dt * c[rk + 1])
            self.pdes["T"].solve_step(rk)

        # update checkpoint and data files
        self.checkpoint.update(t, tstep)
        if tstep % self.params.modsave == 0:
            self.tofile(tstep)

        # update outputs and time
        self.outputs()
        return t + self.params.dt, tstep + 1

    def clean(self) -> None:
        self.TT.destroy()
        self.TB.destroy()
        self.TD.destroy()
        self.TC.destroy()
        self.TDp.destroy()
        self.BD.destroy()
        self.CD.destroy()
