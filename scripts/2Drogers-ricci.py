from firedrake import (
    Constant,
    DirichletBC,
    dx,
    exp,
    grad,
    Function,
    FunctionSpace,
    PETSc,
    SpatialCoordinate,
    split,
    sqrt,
    TestFunctions,
    VTKFile,
)

from common import (
    nl_solve_setup,
    phi_solve_setup,
    poisson_bracket,
    read_rr_config,
    rr_DG_upwind_term,
    rr_src_term,
    rr_SU_term,
    rr_steady_state,
    set_up_mesh,
)
from irksome import Dt
import os.path
from pyop2.mpi import COMM_WORLD
import time


def exp_T_term(T, phi, cfg, eps=1e-2):
    e = Constant(cfg["normalised"]["e"])
    Lambda = Constant(cfg["physical"]["Lambda"])
    return exp(Lambda - e * phi / sqrt(T * T + eps * eps))


def get_phi_bc_ufl(phi, phi_target, t, cfg):
    t_relax = cfg["time"].get("phi_t_relax")
    weight = exp(-(float(t) % t_relax) / t_relax)
    return weight * phi + (1 - weight) * phi_target


def update_phi_bc_funcs(lowx, highx, lowy, highy, phi, grad_phi_x, grad_phi_y, t, cfg):
    delta_x = cfg["mesh"]["dx"]  # / cfg["numerics"]["fe_order"]["phi"]
    # Currently forcing dy=dx...
    delta_y = delta_x

    low_x_target = phi + grad_phi_x * delta_x
    high_x_target = phi - grad_phi_x * delta_x
    low_y_target = phi + grad_phi_y * delta_y
    high_y_target = phi - grad_phi_y * delta_y
    lowx.set_value(get_phi_bc_ufl(phi, low_x_target, t, cfg))
    highx.set_value(get_phi_bc_ufl(phi, high_x_target, t, cfg))
    lowy.set_value(get_phi_bc_ufl(phi, low_y_target, t, cfg))
    highy.set_value(get_phi_bc_ufl(phi, high_y_target, t, cfg))


def gen_phi_bcs(phi_space, phi, grad_phi_x, grad_phi_y, t, cfg):
    lbls = dict(low_x=1, high_x=2, low_y=3, high_y=4)
    t_relax = cfg["time"].get("phi_t_relax")
    if t_relax is None:
        transverse_bdy_lbls = list(lbls.values())
        return [DirichletBC(phi_space, 0.0, transverse_bdy_lbls)]
    else:
        low_x_bc = Function(phi_space, name="low_x")
        high_x_bc = Function(phi_space, name="high_x")
        low_y_bc = Function(phi_space, name="low_y")
        high_y_bc = Function(phi_space, name="high_y")

        # delta_x = cfg["mesh"]["dx"]  # / cfg["numerics"]["fe_order"]["phi"]
        # delta_y = delta_x
        # low_x_bc = Function(get_phi_bc_ufl(phi, phi + grad_phi_x * delta_x, t, cfg))
        # high_x_bc = Function(get_phi_bc_ufl(phi, phi - grad_phi_x * delta_x, t, cfg))
        # low_y_bc = Function(get_phi_bc_ufl(phi, phi + grad_phi_y * delta_y, t, cfg))
        # high_y_bc = Function(get_phi_bc_ufl(phi, phi - grad_phi_y * delta_y, t, cfg))
        funcs = [low_x_bc, high_x_bc, low_y_bc, high_y_bc]
        bcs = [
            DirichletBC(phi_space, low_x_bc, lbls["low_x"]),
            DirichletBC(phi_space, high_x_bc, lbls["high_x"]),
            DirichletBC(phi_space, low_y_bc, lbls["low_y"]),
            DirichletBC(phi_space, high_y_bc, lbls["high_y"]),
        ]
        update_phi_bc_funcs(
            *bcs,
            phi,
            grad_phi_x,
            grad_phi_y,
            t,
            cfg,
        )
        return funcs, bcs


def rogers_ricci2D():
    start = time.time()

    # Read config file (expected next to this script)
    cfg = read_rr_config("2Drogers-ricci_config.yml")
    # Generate mesh
    mesh = set_up_mesh(cfg)
    x, y = SpatialCoordinate(mesh)

    time_cfg = cfg["time"]
    t = Constant(time_cfg["t_start"])
    t_end = time_cfg["t_end"]
    dt = Constant(time_cfg["t_end"] / time_cfg["num_steps"])

    # Function spaces
    DG_or_CG = cfg["numerics"]["discretisation"]
    n_space = FunctionSpace(mesh, DG_or_CG, cfg["numerics"]["fe_order"]["n"])  # n
    w_space = FunctionSpace(mesh, DG_or_CG, cfg["numerics"]["fe_order"]["w"])  # w
    T_space = FunctionSpace(mesh, DG_or_CG, cfg["numerics"]["fe_order"]["T"])  # T
    phi_space = FunctionSpace(mesh, "CG", cfg["numerics"]["fe_order"]["phi"])  # phi

    # Functions (combine time-evolved function spaces to facilitate interaction with Irksome)
    phi = Function(phi_space)
    phi.rename("potential")
    combined_space = n_space * w_space * T_space
    time_evo_funcs = Function(combined_space)
    n, w, T = split(time_evo_funcs)
    subspace_indices = dict(n=0, w=1, T=2)

    # Rename fields and set up funcs for output
    subspace_names = dict(
        n="density",
        w="vorticity",
        T="temperature",
    )
    for fld in subspace_indices.keys():
        time_evo_funcs.sub(subspace_indices[fld]).rename(subspace_names[fld])

    # Source functions
    n_src = rr_src_term(n_space, x, y, "n", cfg)
    T_src = rr_src_term(T_space, x, y, "T", cfg)

    phi_for_bcs = Function(phi_space, name="phi_for_BCs")
    phi_for_bcs.assign(phi)
    grad_phi_x_for_bcs = Function(phi_space, name="grad_phi_x_for_BCs")
    grad_phi_x_for_bcs.interpolate(grad(phi_for_bcs)[0])
    grad_phi_y_for_bcs = Function(phi_space, name="grad_phi_y_for_BCs")
    grad_phi_y_for_bcs.interpolate(grad(phi_for_bcs)[1])
    phi_bc_funcs, phi_bcs = gen_phi_bcs(
        phi_space, phi_for_bcs, grad_phi_x_for_bcs, grad_phi_y_for_bcs, t, cfg
    )
    phi_solver = phi_solve_setup(phi_space, phi, w, cfg, bcs=phi_bcs)

    # Assemble variational problem
    n_test, w_test, T_test = TestFunctions(combined_space)

    sigma_cs_over_R = Constant(
        cfg["normalised"]["sigma"] * cfg["normalised"]["c_s0"] / cfg["normalised"]["R"]
    )
    one_over_B = Constant(1 / cfg["normalised"]["B"])
    h_SU = cfg["mesh"]["Lx"] / cfg["mesh"]["nx"]
    isDG = cfg["numerics"]["discretisation"] == "DG"

    n_terms = (
        (
            Dt(n)
            - one_over_B * poisson_bracket(phi, n)
            + sigma_cs_over_R * n * exp_T_term(T, phi, cfg)
            - n_src
        )
        * n_test
        * dx
    )
    if isDG:
        n_terms += rr_DG_upwind_term(n, n_test, phi, mesh, cfg)
    elif cfg["numerics"]["do_streamline_upwinding"]:
        n_terms += rr_SU_term(n, n_test, phi, h_SU, cfg)

    e = cfg["normalised"]["e"]
    m_i = cfg["normalised"]["m_i"]
    Omega_ci = cfg["normalised"]["omega_ci"]
    w_terms = (
        (
            Dt(w)
            - one_over_B * poisson_bracket(phi, w)
            - Constant(sigma_cs_over_R * m_i * Omega_ci * Omega_ci / e)
            * (1 - exp_T_term(T, phi, cfg))
        )
        * w_test
        * dx
    )
    if isDG:
        w_terms += rr_DG_upwind_term(w, w_test, phi, mesh, cfg)
    elif cfg["numerics"]["do_streamline_upwinding"]:
        w_terms += rr_SU_term(w, w_test, phi, h_SU, cfg)

    T_terms = (
        (
            Dt(T)
            - one_over_B * poisson_bracket(phi, T)
            + Constant(sigma_cs_over_R * 2 / 3)
            * T
            * (1.71 * exp_T_term(T, phi, cfg) - 0.71)
            - T_src
        )
        * T_test
        * dx
    )
    if isDG:
        T_terms += rr_DG_upwind_term(T, T_test, phi, mesh, cfg)
    elif cfg["numerics"]["do_streamline_upwinding"]:
        T_terms += rr_SU_term(T, T_test, phi, h_SU, cfg)

    F = n_terms + w_terms + T_terms

    # Set ICs
    if cfg["model"]["start_from_steady_state"]:
        n_init, T_init, w_init = rr_steady_state(x, y, cfg)
    else:
        n_init = cfg["normalised"]["n_init"]
        T_init = cfg["normalised"]["T_init"]
        w_init = 0.0

    time_evo_funcs.sub(subspace_indices["n"]).interpolate(n_init)
    time_evo_funcs.sub(subspace_indices["T"]).interpolate(T_init)
    time_evo_funcs.sub(subspace_indices["w"]).interpolate(w_init)

    stepper = nl_solve_setup(F, t, dt, time_evo_funcs, cfg)

    # Set up output
    outfile = VTKFile(os.path.join(cfg["root_dir"], cfg["output_base"] + ".pvd"))

    # phi BCs output
    phi_bcs_outfile = VTKFile(os.path.join(cfg["root_dir"], "phi_bc.pvd"))

    PETSc.Sys.Print("\nTimestep loop:")
    step = 0

    twall_last_info = time.time()
    while float(t) < float(t_end):
        if (float(t) + float(dt)) > t_end:
            dt.assign(t_end - float(t))
            PETSc.Sys.Print(f"  Last dt = {dt}")
        phi_for_bcs.assign(phi)
        grad_phi_x_for_bcs.interpolate(grad(phi_for_bcs)[0])
        grad_phi_y_for_bcs.interpolate(grad(phi_for_bcs)[1])
        update_phi_bc_funcs(
            *phi_bcs, phi_for_bcs, grad_phi_x_for_bcs, grad_phi_y_for_bcs, t, cfg
        )
        phi_solver.solve()

        # Write fields on output steps
        if step % cfg["time"]["output_freq"] == 0:
            outfile.write(
                time_evo_funcs.sub(subspace_indices["n"]),
                time_evo_funcs.sub(subspace_indices["w"]),
                time_evo_funcs.sub(subspace_indices["T"]),
                phi,
            )
            phi_bcs_outfile.write(*phi_bc_funcs)

        stepper.advance()
        t.assign(float(t) + float(dt))
        if step % cfg["time"]["info_freq"] == 0:
            dtwall_last_info = time.time() - twall_last_info
            twall_last_info = time.time()
            last_info_step = step + 1 - cfg["time"]["info_freq"]
            if cfg["time"]["info_freq"] == 1 or last_info_step < 0:
                iters_str = f"Iter {step+1:d}"
            else:
                iters_str = f"Iters {last_info_step:d}-{step:d}"
            PETSc.Sys.Print(
                f"  {iters_str}(/{time_cfg['num_steps']:d}) took {dtwall_last_info:.5g} s"
            )
            PETSc.Sys.Print(f"t = {float(t):.5g}")
        step += 1

    wall_time = time.time() - start
    PETSc.Sys.Print("\nDone.")
    PETSc.Sys.Print(f"Total wall time: {wall_time:.5g}")


if __name__ == "__main__":
    rogers_ricci2D()
