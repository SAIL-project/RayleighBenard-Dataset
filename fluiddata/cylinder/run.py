from omegaconf import DictConfig

from fluiddata.utils.callbacks import CylinderVisCallback

try:
    import hydrogym.firedrake as hgym
except ImportError:
    print("Hydrogym/Firedrake not found, Cylinder simulation is not available")


def log(flow: hgym.RotaryCylinder):
    CL, CD = flow.get_observations()
    return CL, CD


def run_cylinder(sim: DictConfig, interval: int):
    # Define system
    flow = hgym.RotaryCylinder(
        Re=sim.re,
        mesh=sim.mesh,
        velocity_order=sim.velocity_order,
    )

    # Callbacks
    print_fmt = "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}"
    callbacks = [
        hgym.utils.io.LogCallback(
            postprocess=log, nvals=2, print_fmt=print_fmt, interval=interval
        ),
        CylinderVisCallback(interval=interval),
    ]

    # Run simulation
    hgym.integrate(
        flow,
        t_span=(0, sim.episode_length),
        dt=sim.dt,
        callbacks=callbacks,
        stabilization=sim.stabilization,
    )
