from firedrake import (
    as_vector,
    Constant,
    DirichletBC,
    dot,
    dS,
    dx,
    exp,
    FacetNormal,
    Function,
    FunctionSpace,
    grad,
    inner,
    LinearVariationalProblem,
    LinearVariationalSolver,
    PETSc,
    SpatialCoordinate,
    split,
    sqrt,
    TestFunction,
    TestFunctions,
    TrialFunction,
    VTKFile,
)

from common import read_rr_config, rr_src_term, rr_steady_state, set_up_mesh
from irksome import Dt, GaussLegendre, TimeStepper
import os.path
from pyop2.mpi import COMM_WORLD
import time


def drift_vel(phi, cfg):
    one_over_B = Constant(1 / cfg["normalised"]["B"])
    return as_vector([-one_over_B * grad(phi)[1], one_over_B * grad(phi)[0]])


def DG_flux_term(tri, test, phi, mesh, cfg):
    vExB = drift_vel(phi, cfg)
    norms = FacetNormal(mesh)
    vExB_n = 0.5 * (dot(vExB, norms) + abs(dot(vExB, norms)))
    return (
        vExB_n("-") * (tri("-") - tri("+")) * test("-") * dS
        + vExB_n("+") * (tri("+") - tri("-")) * test("+") * dS
    )


def exp_T_term(T, phi, cfg, eps=1e-2):
    e = Constant(cfg["normalised"]["e"])
    Lambda = Constant(cfg["physical"]["Lambda"])
    return exp(Lambda - e * phi / sqrt(T * T + eps * eps))


def SU_term(tri, test, phi, h, cfg, eps=1e-2):
    driftvel = drift_vel(phi, cfg)
    return (
        0.5
        * h
        * (dot(driftvel, grad(tri)))
        * dot(driftvel, grad(test))
        * (1 / sqrt((driftvel[0]) ** 2 + (driftvel[1]) ** 2 + eps * eps))
        * dx
    )


def nl_solve_setup(F, t, dt, state, cfg):
    butcher_tableau = GaussLegendre(cfg["time"]["order"])
    nl_solver_params = {
        "snes_max_it": 100,
        "snes_linesearch_type": "l2",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "mat_type": "aij",
        "pc_factor_mat_solver_type": "mumps",
    }
    if cfg["debug"]:
        nl_solver_params["ksp_monitor"] = None
        nl_solver_params["snes_monitor"] = None

    return TimeStepper(F, butcher_tableau, t, dt, state, solver_parameters=nl_solver_params)  # fmt: skip


def phi_solve_setup(phi_space, phi, w, cfg, bcs=None):
    phi_test = TestFunction(phi_space)
    phi_tri = TrialFunction(phi_space)

    rhs_fac = (
        cfg["normalised"]["e"]
        * cfg["normalised"]["B"] ** 2
        / cfg["normalised"]["m_i"]
        / cfg["normalised"]["n_char"]
    )
    # N.B. Integration by parts gives you a -ve sign on the LHS
    Lphi = -inner(grad(phi_test), grad(phi_tri)) * dx
    Rphi = Constant(rhs_fac) * w * phi_test * dx

    # D0 on all boundaries
    if bcs is None:
        if cfg["mesh"]["type"] in ["circle", "cuboid", "rectangle"]:
            bdy_lbl_all = "on_boundary"
        elif cfg["mesh"]["type"] == "cylinder":
            bdy_lbl_all = ("on_boundary", "top", "bottom")
        bcs = DirichletBC(phi_space, 0, bdy_lbl_all)

    phi_problem = LinearVariationalProblem(Lphi, Rphi, phi, bcs=bcs)
    solver_params = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    return LinearVariationalSolver(phi_problem, solver_parameters=solver_params)


def phi_bcs2D(phi_space, T, cfg):
    # Try and get boundary label from mesh cfg; default to best guess if it's not set
    transverse_bdy_lbl = cfg["mesh"].get("transverse_bdy_lbl", "on_boundary")

    # Set fixed boundary value; defaults to zero
    fixed_bdy_val = cfg["numerics"].get("phi_boundary_value", 0.0)
    return DirichletBC(phi_space, fixed_bdy_val, transverse_bdy_lbl)


def poisson_bracket(f, phi):
    return phi.dx(0) * f.dx(1) - phi.dx(1) * f.dx(0)


def rogers_ricci2D():
    start = time.time()

    # Read config file (expected next to this script)
    cfg = read_rr_config("2Drogers-ricci_config.yml")
    # Generate mesh
    mesh = set_up_mesh(cfg)
    x, y = SpatialCoordinate(mesh)

    # Function spaces
    DG_or_CG = cfg["numerics"]["discretisation"]
    n_space = FunctionSpace(mesh, DG_or_CG, 1)  # n
    w_space = FunctionSpace(mesh, DG_or_CG, 1)  # w
    T_space = FunctionSpace(mesh, DG_or_CG, 1)  # T
    phi_space = FunctionSpace(mesh, "CG", 1)  # phi

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

    phi_solver = phi_solve_setup(
        phi_space, phi, w, cfg, bcs=phi_bcs2D(phi_space, T, cfg)
    )

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
            - one_over_B * poisson_bracket(n, phi)
            + sigma_cs_over_R * n * exp_T_term(T, phi, cfg)
            - n_src
        )
        * n_test
        * dx
    )
    if isDG:
        n_terms += DG_flux_term(n, n_test, phi, mesh, cfg)
    elif cfg["numerics"]["do_streamline_upwinding"]:
        n_terms += SU_term(n, n_test, phi, h_SU, cfg)

    e = cfg["normalised"]["e"]
    m_i = cfg["normalised"]["m_i"]
    Omega_ci = cfg["normalised"]["omega_ci"]
    w_terms = (
        (
            Dt(w)
            - one_over_B * poisson_bracket(w, phi)
            - Constant(sigma_cs_over_R * m_i * Omega_ci * Omega_ci / e)
            * (1 - exp_T_term(T, phi, cfg))
        )
        * w_test
        * dx
    )
    if isDG:
        w_terms += DG_flux_term(w, w_test, phi, mesh, cfg)
    elif cfg["numerics"]["do_streamline_upwinding"]:
        w_terms += SU_term(w, w_test, phi, h_SU, cfg)

    T_terms = (
        (
            Dt(T)
            - one_over_B * poisson_bracket(T, phi)
            + Constant(sigma_cs_over_R * 2 / 3)
            * T
            * (1.71 * exp_T_term(T, phi, cfg) - 0.71)
            - T_src
        )
        * T_test
        * dx
    )
    if isDG:
        T_terms += DG_flux_term(T, T_test, phi, mesh, cfg)
    elif cfg["numerics"]["do_streamline_upwinding"]:
        T_terms += SU_term(T, T_test, phi, h_SU, cfg)

    F = n_terms + w_terms + T_terms

    time_cfg = cfg["time"]
    t = Constant(time_cfg["t_start"])
    t_end = time_cfg["t_end"]
    dt = Constant(time_cfg["t_end"] / time_cfg["num_steps"])

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

    PETSc.Sys.Print("\nTimestep loop:")
    step = 0

    twall_last_info = time.time()
    while float(t) < float(t_end):
        if (float(t) + float(dt)) > t_end:
            dt.assign(t_end - float(t))
            PETSc.Sys.Print(f"  Last dt = {dt}")
        phi_solver.solve()

        # Write fields on output steps
        if step % cfg["time"]["output_freq"] == 0:
            outfile.write(
                time_evo_funcs.sub(subspace_indices["n"]),
                time_evo_funcs.sub(subspace_indices["w"]),
                time_evo_funcs.sub(subspace_indices["T"]),
                phi,
            )

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
