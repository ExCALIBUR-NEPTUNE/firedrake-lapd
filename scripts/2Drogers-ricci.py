from firedrake import (
    Constant,
    derivative,
    DirichletBC,
    dx,
    exp,
    Function,
    FunctionSpace,
    grad,
    inner,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    PETSc,
    SpatialCoordinate,
    split,
    sqrt,
    TestFunctions,
    VTKFile,
)

from common import (
    poisson_bracket,
    read_rr_config,
    rr_DG_upwind_term,
    rr_src_term,
    rr_SU_term,
    rr_steady_state,
    set_up_mesh,
)
import os.path
from pyop2.mpi import COMM_WORLD
import time


def setup_nl_solver(eqn, U1, Jp, bcs):
    nl_prob = NonlinearVariationalProblem(eqn, U1, Jp=Jp, bcs=bcs)
    nl_params = {
        # "snes_monitor": None,
        # "snes_converged_reason": None,
        "ksp_type": "gmres",
        # "ksp_converged_reason": None,
        # "snes_atol": 1.0e-50,
        # "snes_stol": 1.0e-50,
        # "snes_rtol": 1.0e-7,
        # "ksp_atol": 1.0e-50,
        # "ksp_rtol": 1.0e-9,
        # "ksp_max_it": 50,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "pc_fieldsplit_0_fields": "0,1,2",
        "pc_fieldsplit_1_fields": "3",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "bjacobi",
        "fieldsplit_0_sub_pc_type": "ilu",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_ksp_reuse_preconditioner": None,
        "fieldsplit_1_pc_type": "lu",
        "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
    }
    return NonlinearVariationalSolver(nl_prob, solver_parameters=nl_params)


def exp_T_term(T, phi, cfg, eps=1e-2):
    e = Constant(cfg["normalised"]["e"])
    Lambda = Constant(cfg["physical"]["Lambda"])
    return exp(Lambda - e * phi / sqrt(T * T + eps * eps))


def lhs_term(start, end, test):
    return inner(end - start, test) * dx


def rogers_ricci2D():
    start = time.time()

    # Read config file (expected next to this script)
    cfg = read_rr_config("2Drogers-ricci_config.yml")
    # Generate mesh
    mesh = set_up_mesh(cfg)
    x, y = SpatialCoordinate(mesh)

    # Function spaces
    DG_or_CG = cfg["numerics"]["discretisation"]
    n_space = FunctionSpace(mesh, DG_or_CG, cfg["numerics"]["fe_order"]["n"])  # n
    w_space = FunctionSpace(mesh, DG_or_CG, cfg["numerics"]["fe_order"]["w"])  # w
    T_space = FunctionSpace(mesh, DG_or_CG, cfg["numerics"]["fe_order"]["T"])  # T
    phi_space = FunctionSpace(mesh, "CG", cfg["numerics"]["fe_order"]["phi"])  # phi

    combined_space = n_space * w_space * T_space * phi_space
    state0 = Function(combined_space)
    state1 = Function(combined_space)
    n0, w0, T0, phi0 = split(state0)
    n1, w1, T1, phi1 = split(state1)
    subspace_indices = dict(n=0, w=1, T=2, phi=3)

    n_avg = (n0 + n1) / 2
    w_avg = (w0 + w1) / 2
    T_avg = (T0 + T1) / 2
    phi_avg = phi1

    # Rename fields and set up funcs for output
    subspace_names = dict(n="density", w="vorticity", T="temperature", phi="potential")
    for fld in subspace_indices.keys():
        state0.sub(subspace_indices[fld]).rename(subspace_names[fld])

    # Time step
    time_cfg = cfg["time"]
    t = Constant(time_cfg["t_start"])
    t_end = time_cfg["t_end"]
    dt = Constant(time_cfg["t_end"] / time_cfg["num_steps"])

    # Source functions
    n_src = rr_src_term(n_space, x, y, "n", cfg)
    T_src = rr_src_term(T_space, x, y, "T", cfg)

    # Assemble variational problem
    n_test, w_test, T_test, phi_test = TestFunctions(combined_space)

    sigma_cs_over_R = Constant(
        cfg["normalised"]["sigma"] * cfg["normalised"]["c_s0"] / cfg["normalised"]["R"]
    )
    one_over_B = Constant(1 / cfg["normalised"]["B"])
    h_SU = cfg["mesh"]["Lx"] / cfg["mesh"]["nx"]
    isDG = cfg["numerics"]["discretisation"] == "DG"

    n_terms = (
        lhs_term(n0, n1, n_test)
        + dt
        * (
            -one_over_B * poisson_bracket(phi_avg, n_avg)
            + sigma_cs_over_R * n_avg * exp_T_term(T_avg, phi_avg, cfg)
            - n_src
        )
        * n_test
        * dx
    )
    if isDG:
        n_terms += dt * rr_DG_upwind_term(n_avg, n_test, phi_avg, mesh, cfg)
    elif cfg["numerics"]["do_streamline_upwinding"]:
        n_terms += dt * rr_SU_term(n_avg, n_test, phi_avg, h_SU, cfg)

    e = cfg["normalised"]["e"]
    m_i = cfg["normalised"]["m_i"]
    Omega_ci = cfg["normalised"]["omega_ci"]
    w_terms = lhs_term(w0, w1, w_test) + dt * (
        (
            -one_over_B * poisson_bracket(phi_avg, w_avg)
            - Constant(sigma_cs_over_R * m_i * Omega_ci * Omega_ci / e)
            * (1 - exp_T_term(T_avg, phi_avg, cfg))
        )
        * w_test
        * dx
    )
    if isDG:
        w_terms += dt * rr_DG_upwind_term(w_avg, w_test, phi_avg, mesh, cfg)
    elif cfg["numerics"]["do_streamline_upwinding"]:
        w_terms += dt * rr_SU_term(w_avg, w_test, phi_avg, h_SU, cfg)

    T_terms = (
        lhs_term(T0, T1, T_test)
        + dt
        * (
            -one_over_B * poisson_bracket(phi_avg, T_avg)
            + Constant(sigma_cs_over_R * 2 / 3)
            * T_avg
            * (1.71 * exp_T_term(T_avg, phi_avg, cfg) - 0.71)
            - T_src
        )
        * T_test
        * dx
    )
    if isDG:
        T_terms += dt * rr_DG_upwind_term(T_avg, T_test, phi_avg, mesh, cfg)
    elif cfg["numerics"]["do_streamline_upwinding"]:
        T_terms += dt * rr_SU_term(T_avg, T_test, phi_avg, h_SU, cfg)

    phi_terms = (
        grad(phi_avg)[0] * grad(phi_test)[0] + grad(phi_avg)[1] * grad(phi_test)[1]
    ) * dx + w_avg * phi_test * dx

    F = w_terms + T_terms + phi_terms
    if cfg["model"]["evolve_density"]:
        F += n_terms
    F_for_jacobian = F + phi_avg * phi_test * dx
    Jp = derivative(F_for_jacobian, state1)

    bcs = DirichletBC(combined_space.sub(subspace_indices["phi"]), 0.0, "on_boundary")
    nl_solver = setup_nl_solver(F, state1, Jp, bcs)

    # Set ICs
    if cfg["model"]["start_from_steady_state"]:
        n_init, T_init, w_init = rr_steady_state(x, y, cfg)
    else:
        n_init = cfg["normalised"]["n_init"]
        T_init = cfg["normalised"]["T_init"]
        w_init = 0.0

    state0.sub(subspace_indices["n"]).interpolate(n_init)
    state0.sub(subspace_indices["T"]).interpolate(T_init)
    state0.sub(subspace_indices["w"]).interpolate(w_init)

    # Set up output
    outfile = VTKFile(os.path.join(cfg["root_dir"], cfg["output_base"] + ".pvd"))

    PETSc.Sys.Print("\nTimestep loop:")
    step = 0

    state1.assign(state0)

    twall_last_info = time.time()
    while float(t) < float(t_end):
        if (float(t) + float(dt)) > t_end:
            dt.assign(t_end - float(t))
            PETSc.Sys.Print(f"  Last dt = {dt}")
        t.assign(float(t) + float(dt))
        nl_solver.solve()
        state0.assign(state1)

        # Write fields on output steps
        if step % cfg["time"]["output_freq"] == 0:
            outfile.write(
                state0.sub(subspace_indices["n"]),
                state0.sub(subspace_indices["w"]),
                state0.sub(subspace_indices["T"]),
                state0.sub(subspace_indices["phi"]),
            )

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
