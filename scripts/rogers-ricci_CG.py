from common import read_rr_config, rr_src_term, set_up_mesh
from firedrake import (
    as_vector,
    Constant,
    DirichletBC,
    dx,
    exp,
    Function,
    FunctionSpace,
    grad,
    inner,
    PETSc,
    solve,
    SpatialCoordinate,
    split,
    TestFunction,
    TestFunctions,
    TrialFunction,
    VectorSpaceBasis,
    VTKFile,
)
from irksome import Dt, GaussLegendre, TimeStepper
import os.path
from pyop2.mpi import COMM_WORLD
import time


def su_term(h, tri, test, vel_long, vel_eps=0.001):
    vel = as_vector([0, 0, vel_long])
    return (
        0.5
        * h
        * (
            inner(vel, grad(test))
            * inner(vel, grad(tri))
            / (inner(vel, vel) + vel_eps * vel_eps)
        )
        * dx
    )


def poisson_bracket(f, phi, c_over_B):
    return Constant(c_over_B) * (phi.dx(0) * f.dx(1) - phi.dx(1) * f.dx(0))


def nl_solve_setup(F, t, dt, n_ui_ue_T_w, bcs, cfg):
    butcher_tableau = GaussLegendre(cfg["order"])
    nl_solver_params = {
        "snes_monitor": None,
        "snes_max_it": 100,
        "snes_linesearch_type": "l2",
        "snes_atol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "mat_type": "aij",
        "pc_factor_mat_solver_type": "mumps",
    }
    return TimeStepper(F, butcher_tableau, t, dt, n_ui_ue_T_w, solver_parameters=nl_solver_params, bcs=bcs)  # fmt: skip


def phi_solve_setup(phi_space, vorticity, mesh_cfg):
    phi_test = TestFunction(phi_space)
    phi_tri = TrialFunction(phi_space)
    Lphi = (
        grad(phi_tri)[0] * grad(phi_test)[0] + grad(phi_tri)[1] * grad(phi_test)[1]
    ) * dx
    Rphi = vorticity * phi_test * dx

    # D0 on all boundaries
    if mesh_cfg["type"] == "cuboid":
        bdy_lbl_all = "on_boundary"
    elif mesh_cfg["type"] == "cylinder":
        bdy_lbl_all = ("on_boundary", "top", "bottom")
    phi_BCs = DirichletBC(phi_space, 0, bdy_lbl_all)

    # Solver params
    solver_params = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "mat_type": "aij",
        "pc_factor_mat_solver_type": "mumps",
    }

    nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)

    return Lphi == Rphi, phi_BCs, solver_params, nullspace


def rogers_ricci():
    start = time.time()

    # Read config file (expected next to this script)
    cfg = read_rr_config("rogers-ricci_config.yml")
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

    phi_eqn, phi_bcs, phi_solve_params, nullspace = phi_solve_setup(
        phi_space, w, cfg["mesh"]
    )

    # Assemble variational problem
    if is_isothermal:
        n_test, ui_test, ue_test, w_test = TestFunctions(combined_space)
    else:
        n_test, ui_test, ue_test, T_test, w_test = TestFunctions(combined_space)

    # h factor for streamline-upwinding
    norm_cfg = cfg["normalised"]
    do_SU = cfg["numerics"]["do_streamline_upwinding"]
    if do_SU:
        h_long = norm_cfg["dz"]

    # fmt: off
    n_terms = (
        Dt(n) * n_test * dx
        + (grad(n * ue)[2] * n_test) * dx
        - poisson_bracket(n, phi, norm_cfg["c_over_B"]) * n_test * dx
        - (n_src * n_test) * dx
    )
    if do_SU: 
        n_terms += su_term(h_long, n, n_test, ue)

    ui_terms = (
        Dt(ui) * ui_test * dx
        - poisson_bracket(ui, phi, norm_cfg["c_over_B"]) * ui_test * dx
        + (ui * grad(ui)[2] * ui_test) * dx
        + (grad(n * T)[2] / n * ui_test) * dx
    )
    if do_SU: 
        ui_terms += su_term(h_long, ui, ui_test, ui)

    m_e = norm_cfg["m_e"]
    charge_e = norm_cfg["e"]
    j_par = charge_e * n * (ui - ue)
    sigma_par = norm_cfg["sigma_par"]
    ue_terms = (
        m_e * Dt(ue) * ue_test * dx
        - poisson_bracket(ue, phi, norm_cfg["c_over_B"]) * ue_test * dx
        + (m_e * ue * grad(ue)[2] * ue_test) * dx
        + (T / n * grad(n)[2] * ue_test) * dx
        - (charge_e * grad(phi)[2] * ue_test) * dx
        - (charge_e * j_par / sigma_par * ue_test) * dx
    )
    if do_SU: 
        ue_terms += su_term(h_long, m_e*ue, ue_test, ue)

    if is_isothermal:
        T_terms = 0
    else:
        ue_terms += (1.71 * grad(T)[2] * ue_test) * dx
        T_terms = (
            Dt(T) * T_test * dx
            - poisson_bracket(T, phi, norm_cfg["c_over_B"]) * T_test * dx
            - (2.0 / 3 * T / charge_e / n * 0.71 * grad(j_par)[2] * T_test) * dx
            + (2.0 / 3 * T * grad(ue)[2] * T_test) * dx
            + (ue * grad(T)[2] * T_test) * dx
            - (T_src * T_test) * dx
        )
        if do_SU: 
            T_terms += su_term(h_long, T, T_test, ue)

    Omega_ci = norm_cfg["omega_ci"]
    m_i = norm_cfg["m_i"]
    w_terms = (
        Dt(w) * w_test * dx
        - poisson_bracket(w, phi, norm_cfg["c_over_B"]) * w_test * dx
        + (ui * grad(w)[2] * w_test) * dx
        - (m_i * Omega_ci * Omega_ci / charge_e / charge_e / n * grad(j_par)[2] * w_test) * dx
    )
    if do_SU: 
        w_terms += su_term(h_long, w, w_test, ui)

    # fmt: on
    F = n_terms + ui_terms + ue_terms + T_terms + w_terms

    time_cfg = cfg["time"]
    t = Constant(time_cfg["t_start"])
    t_end = time_cfg["t_end"]
    dt = Constant(time_cfg["t_end"] / time_cfg["num_steps"])

    mesh_cfg = cfg["mesh"]
    # Move to mesh set up?
    if mesh_cfg["type"] == "cuboid":
        par_bdy_lbl_lower = 5
        par_bdy_lbl_upper = 6
    elif mesh_cfg["type"] == "cylinder":
        par_bdy_lbl_lower = "bottom"
        par_bdy_lbl_upper = "top"

    # Set up Bohm BCs for ui,ue
    cs = norm_cfg["u_ref"]
    ui_bcs = [
        DirichletBC(combined_space.sub(subspace_indices["ui"]), -cs, par_bdy_lbl_lower),
        DirichletBC(combined_space.sub(subspace_indices["ui"]), cs, par_bdy_lbl_upper),
    ]
    # Excluding phi, T dependence from ue BCs for now
    coulomb_log = cfg["physical"]["Lambda"]
    if cfg["model"]["coulomb_fac_enabled"]:
        coulomb_fac = exp(coulomb_log - phi / T)
    else:
        coulomb_fac = 1
    ue_bcs = [
        DirichletBC(
            combined_space.sub(subspace_indices["ue"]),
            -cs * coulomb_fac,
            par_bdy_lbl_lower,
        ),
        DirichletBC(
            combined_space.sub(subspace_indices["ue"]),
            cs * coulomb_fac,
            par_bdy_lbl_upper,
        ),
    ]
    bcs = [*ui_bcs, *ue_bcs]

    stepper = nl_solve_setup(F, t, dt, time_evo_funcs, bcs, time_cfg)

    outfile = VTKFile(os.path.join(cfg["root_dir"], cfg["output_base"] + ".pvd"))

    # Initial conditions
    time_evo_funcs.sub(subspace_indices["n"]).interpolate(norm_cfg["n_init"])
    # Ion and electron velocities are initially linear in z
    time_evo_funcs.sub(subspace_indices["ui"]).interpolate(
        2 * norm_cfg["u_ref"] * z / mesh_cfg["Lz"]
    )
    time_evo_funcs.sub(subspace_indices["ue"]).interpolate(
        2 * norm_cfg["u_ref"] * z / mesh_cfg["Lz"]
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
        solve(phi_eqn, phi, nullspace=nullspace, solver_parameters=phi_solve_params, bcs=phi_bcs)  # fmt: skip

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
