from common import (
    nl_solve_setup,
    phi_solve_setup,
    poisson_bracket,
    read_rr_config,
    rr_src_term,
    rr_SU_term,
    set_up_mesh,
)
from firedrake import (
    Constant,
    DirichletBC,
    dx,
    exp,
    Function,
    FunctionSpace,
    grad,
    PETSc,
    SpatialCoordinate,
    split,
    sqrt,
    TestFunctions,
    VTKFile,
)
from irksome import Dt
import os.path
from pyop2.mpi import COMM_WORLD
import time


def gen_bohm_bcs(ui_space, ue_space, phi, T, cfg, T_eps=1e-2):
    """Set up Bohm BCs for ui,ue"""
    cs = cfg["normalised"]["c_s0"]

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
        coulomb_fac = exp(coulomb_log - phi / sqrt(T * T + T_eps * T_eps))
    else:
        coulomb_fac = 1
    ue_bcs = [
        DirichletBC(
            ue_space,
            -cs * coulomb_fac,
            par_bdy_lbl_lower,
        ),
        DirichletBC(
            ue_space,
            cs * coulomb_fac,
            par_bdy_lbl_upper,
        ),
    ]
    return [*ui_bcs, *ue_bcs]


def rogers_ricci():
    start = time.time()

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

    # Functions (combine time-evolved function spaces to facilitate interaction with Irksome)
    phi = Function(phi_space)
    is_isothermal = cfg["model"]["is_isothermal"]
    if is_isothermal:
        combined_space = n_space * ui_space * ue_space * w_space
        time_evo_funcs = Function(combined_space)
        n, ui, ue, w = split(time_evo_funcs)
        T = Constant(cfg["normalised"]["T_init"], name="T")
        subspace_indices = dict(n=0, ui=1, ue=2, w=3)
    else:
        combined_space = n_space * ui_space * ue_space * T_space * w_space
        time_evo_funcs = Function(combined_space)
        n, ui, ue, T, w = split(time_evo_funcs)
        subspace_indices = dict(n=0, ui=1, ue=2, T=3, w=4)

    # Rename fields and set up funcs for output
    subspace_names = dict(
        n="density",
        ui="ion velocity",
        ue="electron velocity",
        T="temperature",
        w="vorticity",
    )
    for fld in subspace_indices.keys():
        time_evo_funcs.sub(subspace_indices[fld]).rename(subspace_names[fld])
    phi.rename("potential")
    output_funcs = [
        time_evo_funcs.sub(subspace_indices[fld]) for fld in subspace_indices.keys()
    ]
    output_funcs.append(phi)

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

    phi_solver = phi_solve_setup(phi_space, phi, w, cfg)

    # Assemble variational problem
    if is_isothermal:
        n_test, ui_test, ue_test, w_test = TestFunctions(combined_space)
    else:
        n_test, ui_test, ue_test, T_test, w_test = TestFunctions(combined_space)

    # h factor for streamline-upwinding
    norm_cfg = cfg["normalised"]
    do_SU = cfg["numerics"]["do_streamline_upwinding"]
    one_over_B = Constant(1 / cfg["normalised"]["B"])
    n_eps = 1e-6
    if do_SU:
        h = cfg["mesh"]["dx"]

    # fmt: off
    n_terms = (
        Dt(n) * n_test * dx
        - one_over_B * poisson_bracket(phi, n) * n_test * dx
        + (grad(n * ue)[2] * n_test) * dx
        - (n_src * n_test) * dx
    )
    if do_SU:
        n_terms += rr_SU_term(n, n_test, phi, h, cfg, vel_par=ue)

    ui_terms = (
        Dt(ui) * ui_test * dx
        - one_over_B * poisson_bracket(phi, ui) * ui_test * dx
        + (ui * grad(ui)[2] * ui_test) * dx
        + (grad(n * T)[2] / sqrt(n * n + n_eps * n_eps) * ui_test) * dx
    )
    if do_SU: 
        ui_terms += rr_SU_term(ui, ui_test, phi, h, cfg, vel_par=ui)

    m_e = norm_cfg["m_e"]
    charge_e = norm_cfg["e"]
    j_par = charge_e * n * (ui - ue)
    sigma_par = norm_cfg["sigma_par"]
    ue_terms = (
        m_e * (Dt(ue) - one_over_B * poisson_bracket(phi, ue)) * ue_test * dx
        + (m_e * ue * grad(ue)[2] * ue_test) * dx
        + (T / n * grad(n)[2] * ue_test) * dx
        - (charge_e * grad(phi)[2] * ue_test) * dx
        - (charge_e * j_par / sigma_par * ue_test) * dx
    )
    if do_SU: 
        ue_terms += rr_SU_term(ue, ue_test, phi, h, cfg, vel_par=ue)

    if is_isothermal:
        T_terms = 0
    else:
        ue_terms += (1.71 * grad(T)[2] * ue_test) * dx
        T_terms = (
            Dt(T) * T_test * dx
            - one_over_B * poisson_bracket(phi, T) * T_test * dx
            - (2.0 / 3 * T / charge_e / sqrt(n * n + n_eps * n_eps) * 0.71 * grad(j_par)[2] * T_test) * dx
            + (2.0 / 3 * T * grad(ue)[2] * T_test) * dx
            + (ue * grad(T)[2] * T_test) * dx
            - (T_src * T_test) * dx
        )
        if do_SU: 
            T_terms += rr_SU_term(T, T_test, phi, h, cfg, vel_par=ue)

    Omega_ci = norm_cfg["omega_ci"]
    m_i = norm_cfg["m_i"]
    w_terms = (
        Dt(w) * w_test * dx
        - one_over_B * poisson_bracket(phi, w) * w_test * dx
        + (ui * grad(w)[2] * w_test) * dx
        - (m_i * Omega_ci * Omega_ci / charge_e / charge_e / sqrt(n * n + n_eps * n_eps) * grad(j_par)[2] * w_test) * dx
    )
    if do_SU: 
        w_terms += rr_SU_term(w, w_test, phi, h, cfg, vel_par=ui)

    # fmt: on
    F = n_terms + ui_terms + ue_terms + T_terms + w_terms

    time_cfg = cfg["time"]
    t = Constant(time_cfg["t_start"])
    t_end = time_cfg["t_end"]
    dt = Constant(time_cfg["t_end"] / time_cfg["num_steps"])

    mesh_cfg = cfg["mesh"]

    bohm_bcs = gen_bohm_bcs(
        combined_space.sub(subspace_indices["ui"]),
        combined_space.sub(subspace_indices["ue"]),
        phi,
        T,
        cfg,
    )

    stepper = nl_solve_setup(F, t, dt, time_evo_funcs, cfg, bcs=bohm_bcs)

    outfile = VTKFile(os.path.join(cfg["root_dir"], cfg["output_base"] + ".pvd"))

    # Initial conditions
    time_evo_funcs.sub(subspace_indices["n"]).interpolate(norm_cfg["n_init"])
    # Ion and electron velocities are initially linear in z
    time_evo_funcs.sub(subspace_indices["ui"]).interpolate(
        2 * norm_cfg["c_s0"] * z / mesh_cfg["Lz"]
    )
    time_evo_funcs.sub(subspace_indices["ue"]).interpolate(
        2 * norm_cfg["c_s0"] * z / mesh_cfg["Lz"]
    )
    if not is_isothermal:
        time_evo_funcs.sub(subspace_indices["T"]).interpolate(norm_cfg["T_init"])
    # Vorticity = 0
    time_evo_funcs.sub(subspace_indices["w"]).interpolate(0)

    outfile.write(*output_funcs)

    PETSc.Sys.Print("\nTimestep loop:")
    step = 0
    while float(t) < float(t_end):
        it_start = time.time()
        if (float(t) + float(dt)) > t_end:
            dt.assign(t_end - float(t))
            PETSc.Sys.Print(f"  Last dt = {dt}")
        phi_solver.solve()

        # Write fields on output steps
        if step % cfg["time"]["output_freq"] == 0:
            outfile.write(*output_funcs)

        stepper.advance()
        t.assign(float(t) + float(dt))
        it_end = time.time()
        it_wall_time = it_end - it_start
        PETSc.Sys.Print(
            f"  Iter {step+1:d}/{time_cfg['num_steps']:d} took {it_wall_time:.5g} s"
        )
        PETSc.Sys.Print(f"t = {float(t):.5g}")
        step += 1
    end = time.time()
    wall_time = end - start

    PETSc.Sys.Print("\nDone.")
    PETSc.Sys.Print(f"Total wall time: {wall_time:.5g}")


if __name__ == "__main__":
    rogers_ricci()
