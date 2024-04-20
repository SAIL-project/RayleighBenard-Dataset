import hydrogym.firedrake as hgym
import psutil

from rbcdata.utils.callbacks import CylinderVisCallback, H5DatasetCallback


def compute_vort(flow):
    return (flow.u, flow.p, flow.vorticity())


def log_postprocess(flow):
    mem_usage = psutil.virtual_memory().percent
    CL, CD = flow.get_observations()
    return CL, CD, mem_usage


def generate_cylinder(cfg):
    # Define system
    flow = hgym.RotaryCylinder(
        Re=cfg.re,
        mesh=cfg.mesh,
        velocity_order=cfg.velocity_order,
    )

    # Callbacks
    print_fmt = "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}\t\t Mem: {3:0.1f}"

    callbacks = [
        hgym.utils.io.LogCallback(
            postprocess=log_postprocess, nvals=3, print_fmt=print_fmt, interval=10
        ),
        #  hgym.io.ParaviewCallback(interval=10, filename="out.pvd", postprocess=compute_vort),
        CylinderVisCallback(interval=10),
        H5DatasetCallback(interval=10),
    ]

    # Run simulation
    hgym.integrate(
        flow,
        t_span=(0, cfg.episode_length),
        dt=cfg.dt,
        callbacks=callbacks,
        stabilization=cfg.stabilization,
    )
