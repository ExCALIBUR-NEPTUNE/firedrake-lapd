from common import (
    poisson_bracket,
    read_rr_config,
    rr_src_term,
    rr_SU_term,
    set_up_mesh,
)
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
import os.path
from pyop2.mpi import COMM_WORLD
import time


def setup_nl_solver(eqn, U1, Jp, bcs, cfg):
    nfields = 5 if cfg["model"]["is_isothermal"] else 6
    potential_idx_str = f"{nfields-1}"
    other_indices_str = ",".join(str(idx) for idx in range(nfields - 1))
    nl_prob = NonlinearVariationalProblem(eqn, U1, Jp=Jp, bcs=bcs)
    nl_params = {
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "pc_fieldsplit_0_fields": other_indices_str,
        "pc_fieldsplit_1_fields": potential_idx_str,
        "fieldsplit_0_ksp_type": "gmres",
        "fieldsplit_0_ksp_gmres_restart": 100,
        "fieldsplit_0_pc_type": "bjacobi",
        "fieldsplit_0_sub_pc_type": "ilu",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_ksp_reuse_preconditioner": None,
        "fieldsplit_1_pc_type": "lu",
        "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
    }
    if cfg["debug_ksp"] or cfg["debug_snes"]:
        nl_params["ksp_converged_reason"] = None
        nl_params["ksp_monitor_true_residual"] = None
    if cfg["debug_ksp"]:
        nl_params["ksp_view"] = None
    if cfg["debug_snes"]:
        nl_params["snes_converged_reason"] = None
        nl_params["snes_monitor"] = None
    return NonlinearVariationalSolver(nl_prob, solver_parameters=nl_params)


def lhs_term(start, end, test):
    return inner(end - start, test) * dx


def gen_bohm_bcs(ui_space, ue_space, phi, T, cfg):
    """Set up Bohm BCs for ui,ue"""
    cs = cfg["normalised"]["c_s0"]
    T_eps = cfg["numerics"]["T_eps"]

    # Move to mesh set up?
    if cfg["mesh"]["type"] == "cuboid":
        par_bdy_lbl_lower = 5
        par_bdy_lbl_upper = 6
    elif cfg["mesh"]["type"] == "cylinder":
        par_bdy_lbl_lower = "bottom"
        par_bdy_lbl_upper = "top"

    ui_bcs = [
        DirichletBC(ui_space, -cs, par_bdy_lbl_lower),
        DirichletBC(ui_space, cs, par_bdy_lbl_upper),
    ]
    # Option to include/exclude phi, T dependence from ue BCs
    coulomb_log = cfg["physical"]["Lambda"]
    if cfg["model"]["coulomb_fac_enabled"]:
        # bohm_expr = exp(coulomb_log - phi / sqrt(T * T + T_eps * T_eps))
        bohm_expr = 1 + coulomb_log - phi / sqrt(T * T + T_eps * T_eps)
    else:
        bohm_expr = 1
    ue_bcs = [
        DirichletBC(
            ue_space,
            -cs * bohm_expr,
            par_bdy_lbl_lower,
        ),
        DirichletBC(
            ue_space,
            cs * bohm_expr,
            par_bdy_lbl_upper,
        ),
    ]

    ue_bc_low = Function(ue_space, name="ue_low")
    ue_bc_low.interpolate(-cs * bohm_expr)
    ue_bc_high = Function(ue_space, name="ue_high")
    ue_bc_high.interpolate(cs * bohm_expr)
    outfile = VTKFile(os.path.join(cfg["root_dir"], "bcs.pvd"))
    outfile.write(ue_bc_low, ue_bc_high)

    return [*ui_bcs, *ue_bcs]


def gen_phi_bcs(phi_space, phi, dt, cfg):
    if cfg["mesh"]["type"] == "cuboid":
        delta_x = cfg["mesh"]["dx"] / cfg["numerics"]["fe_order"]["phi"]
        # Currently forcing dy=dx...
        delta_y = delta_x
    else:
        raise RuntimeError("phi BCs only set up for cuboid domain")
    f_relax = cfg["time"]["phi_t_relax"] / dt
    x_bdy_lbls = [1, 2]
    y_bdy_lbls = [3, 4]

    x_phi_bc = Function(phi_space, name="x_phi_BC")
    x_phi_bc.interpolate(phi - f_relax * grad(phi)[0] * delta_x)
    y_phi_bc = Function(phi_space, name="y_phi_BC")
    y_phi_bc.interpolate(phi - f_relax * grad(phi)[1] * delta_y)
    outfile = VTKFile(os.path.join(cfg["root_dir"], "phi_bc.pvd"))
    outfile.write(x_phi_bc, y_phi_bc)
    #
    return [
        DirichletBC(phi_space, x_phi_bc, x_bdy_lbls),
        DirichletBC(phi_space, y_phi_bc, y_bdy_lbls),
    ]


def rogers_ricci():

    # Read config file (expected next to this script)
    cfg = read_rr_config("rogers-ricci_config.yml")

    assert cfg["numerics"]["discretisation"] == "CG"
    # Generate mesh
    mesh = set_up_mesh(cfg)
    x, y, z = SpatialCoordinate(mesh)

    # Function spaces
    n_space = FunctionSpace(mesh, "CG", 1)  # n
    ui_space = FunctionSpace(mesh, "CG", 1)  # ui - parallel ion velocity
    ue_space = FunctionSpace(mesh, "CG", 1)  # ue - parallel electron velocity
    T_space = FunctionSpace(mesh, "CG", 1)  # T
    w_space = FunctionSpace(mesh, "CG", 1)  # w
    phi_space = FunctionSpace(mesh, "CG", 1)  # phi

    is_isothermal = cfg["model"]["is_isothermal"]
    if is_isothermal:
        combined_space = n_space * ui_space * ue_space * w_space * phi_space
        state0 = Function(combined_space)
        state1 = Function(combined_space)
        n0, ui0, ue0, w0, phi0 = split(state0)
        n1, ui1, ue1, w1, phi1 = split(state1)
        Th = Constant(cfg["normalised"]["T_init"], name="T")
        subspace_indices = dict(n=0, ui=1, ue=2, w=3, phi=4)
    else:
        combined_space = n_space * ui_space * ue_space * T_space * w_space * phi_space
        state0 = Function(combined_space)
        state1 = Function(combined_space)
        n0, ui0, ue0, T0, w0, phi0 = split(state0)
        n1, ui1, ue1, T1, w1, phi1 = split(state1)
        Th = (T0 + T1) / 2
        subspace_indices = dict(n=0, ui=1, ue=2, T=3, w=4, phi=5)
    nh = (n0 + n1) / 2
    uih = (ui0 + ui1) / 2
    ueh = (ue0 + ue1) / 2
    wh = (w0 + w1) / 2
    phih = phi1

    # Rename fields and set up funcs for output
    subspace_names = dict(
        n="density",
        ui="ion velocity",
        ue="electron velocity",
        T="temperature",
        w="vorticity",
        phi="potential",
    )
    for fld in subspace_indices.keys():
        state1.sub(subspace_indices[fld]).rename(subspace_names[fld])
    output_funcs = [
        state1.sub(subspace_indices[fld]) for fld in subspace_indices.keys()
    ]

    # Time setup
    time_cfg = cfg["time"]
    t = Constant(time_cfg["t_start"])
    t_end = time_cfg["t_end"]
    dt = Constant(time_cfg["t_end"] / time_cfg["num_steps"])

    # Source functions
    n_src = rr_src_term(n_space, x, y, "n", cfg)
    if not is_isothermal:
        T_src = rr_src_term(T_space, x, y, "T", cfg)

    # # Check the source functions look ok
    # outfile = VTKFile(f"src_funcs.pvd")
    # if is_isothermal:
    #     outfile.write(n_src)
    # else:
    #     outfile.write(n_src, T_src)

    # Assemble variational problem
    if is_isothermal:
        n_test, ui_test, ue_test, w_test, phi_test = TestFunctions(combined_space)
    else:
        n_test, ui_test, ue_test, T_test, w_test, phi_test = TestFunctions(
            combined_space
        )

    # h factor for streamline-upwinding
    norm_cfg = cfg["normalised"]
    do_SU = cfg["numerics"]["do_streamline_upwinding"]
    one_over_B = Constant(1 / cfg["normalised"]["B"])
    if do_SU:
        h = cfg["mesh"]["dx"]

    # fmt: off
    nh_plus_eps = sqrt(nh * nh + cfg["numerics"]["n_eps"] * cfg["numerics"]["n_eps"])
    n_terms = lhs_term(n0, n1, n_test) + dt * (
        - one_over_B * poisson_bracket(phih, nh) * n_test
        + (grad(nh * ueh)[2] * n_test)
        - (n_src * n_test)
    ) * dx(degree=cfg["numerics"]["quadrature_degree"])
    if do_SU:
        n_terms += dt * rr_SU_term(nh, n_test, phih, h, cfg, vel_par=ueh)

    ui_terms = lhs_term(ui0, ui1, ui_test) + dt * (
        - one_over_B * poisson_bracket(phih, uih) * ui_test
        + (uih * grad(uih)[2] * ui_test)
        + (grad(nh * Th)[2] / nh_plus_eps * ui_test)
    ) * dx(degree=cfg["numerics"]["quadrature_degree"])
    if do_SU: 
        ui_terms += dt * rr_SU_term(uih, ui_test, phih, h, cfg, vel_par=uih)

    charge_e = norm_cfg["e"]
    j_par = charge_e * nh * (uih - ueh)
    tau = cfg["model"]["elec_ion_mass_ratio"]
    nu = cfg["physical"]["nu"]
    ue_terms = lhs_term(ue0, ue1, ue_test) + dt * (
        - one_over_B * poisson_bracket(phih, ueh) * ue_test
        + ueh * grad(ueh)[2] * ue_test
        + tau * Th / nh_plus_eps * grad(nh)[2] * ue_test
        - tau * grad(phih)[2] * ue_test
        - nu*tau*nh*(uih - ueh) * ue_test
    ) * dx(degree=cfg["numerics"]["quadrature_degree"])
    if do_SU: 
        ue_terms += dt * rr_SU_term(ueh, ue_test, phih, h, cfg, vel_par=ueh)

    if is_isothermal:
        T_terms = 0
    else:
        ue_terms += dt * (1.71 * tau * grad(Th)[2] * ue_test) * dx(degree=cfg["numerics"]["quadrature_degree"])
        T_terms = lhs_term(T0, T1, T_test) + dt * (
            - one_over_B * poisson_bracket(phih, Th) * T_test
            - (2.0 / 3 * 0.71 * Th/nh_plus_eps * grad(nh * (uih - ueh))[2] * T_test)
            + (2.0 / 3 * Th * grad(ueh)[2] * T_test) 
            + (ueh * grad(Th)[2] * T_test)
            - (T_src * T_test)
        ) * dx(degree=cfg["numerics"]["quadrature_degree"])
        if do_SU: 
            T_terms += dt * rr_SU_term(Th, T_test, phih, h, cfg, vel_par=ueh)

    Omega_ci = norm_cfg["omega_ci"]
    m_i = norm_cfg["m_i"]
    w_terms = lhs_term(w0, w1, w_test) + dt * (- one_over_B * poisson_bracket(phih, wh) * w_test
        + (uih * grad(wh)[2] * w_test)
        - (1 /nh_plus_eps * grad(nh * (uih - ueh))[2] * w_test)
    ) * dx(degree=cfg["numerics"]["quadrature_degree"])
    if do_SU: 
        w_terms += dt * rr_SU_term(wh, w_test, phih, h, cfg, vel_par=uih)

    phi_terms = (
        grad(phih)[0] * grad(phi_test)[0] + grad(phih)[1] * grad(phi_test)[1]
    ) * dx + wh * phi_test * dx

    # fmt: on
    F = n_terms + ui_terms + ue_terms + T_terms + w_terms + phi_terms
    F_for_jacobian = F + phih * phi_test * dx
    Jp = derivative(F_for_jacobian, state1)

    # Initial conditions
    mesh_cfg = cfg["mesh"]

    # Ion and electron velocities are initially linear in z
    state0.sub(subspace_indices["ui"]).interpolate(
        2 * norm_cfg["c_s0"] * z / mesh_cfg["Lz"]
    )
    state0.sub(subspace_indices["ue"]).interpolate(
        2 * norm_cfg["c_s0"] * z / mesh_cfg["Lz"]
    )

    coulomb_log = cfg["physical"]["Lambda"]
    if cfg["model"]["exp_ics"]:
        r = sqrt(x * x + y * y)
        scale = 80 * cfg["normalised"]["Ls"]
        T_init = 1e-6 * exp(-(r * r) / scale)
        phi_init = coulomb_log * T_init
        n_init = T_init
        w_init = coulomb_log * 4 * T_init * (scale - r * r) / scale / scale / 20
    else:
        n_init = norm_cfg["n_init"]
        T_init = norm_cfg["T_init"]
        phi_init = coulomb_log * T_init
        w_init = 0

    state0.sub(subspace_indices["n"]).interpolate(n_init)
    if not is_isothermal:
        state0.sub(subspace_indices["T"]).interpolate(T_init)
    state0.sub(subspace_indices["w"]).interpolate(w_init)
    state0.sub(subspace_indices["phi"]).interpolate(phi_init)

    # Set up output
    outfile = VTKFile(os.path.join(cfg["root_dir"], cfg["output_base"] + ".pvd"))

    # Timestep logger
    timestep_output_file = os.path.join(cfg["root_dir"], "timesteps.csv")
    if COMM_WORLD.rank == 0:
        with open(timestep_output_file, "w") as timestep_output:
            timestep_output.write(("step, time, duration\n"))

    PETSc.Sys.Print("\nTimestep loop:")
    step = 0
    state1.assign(state0)

    # BCs
    bcs = gen_bohm_bcs(
        combined_space.sub(subspace_indices["ui"]),
        combined_space.sub(subspace_indices["ue"]),
        phih,
        Th,
        cfg,
    )
    bcs.extend(gen_phi_bcs(combined_space.sub(subspace_indices["phi"]), phih, dt, cfg))

    nl_solver = setup_nl_solver(F, state1, Jp, bcs, cfg)

    # Write ICs
    outfile.write(*output_funcs)

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
            outfile.write(*output_funcs)

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

        if COMM_WORLD.rank == 0:
            with open(timestep_output_file, "a") as timestep_output:
                timestep_output.write(f"{step}, {float(t)}, {dtwall_last_info}\n")
        step += 1

    wall_time = time.time() - start
    PETSc.Sys.Print("\nDone.")
    PETSc.Sys.Print(f"Total wall time: {wall_time:.5g}")


if __name__ == "__main__":
    rogers_ricci()
