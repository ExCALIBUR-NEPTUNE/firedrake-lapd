from .io import read_yaml_config, set_default_param
import math
from firedrake import (
    as_vector,
    Constant,
    cosh,
    DirichletBC,
    dot,
    dS,
    dx,
    FacetNormal,
    Function,
    grad,
    LinearVariationalProblem,
    LinearVariationalSolver,
    PETSc,
    sqrt,
    tanh,
    TestFunction,
    TrialFunction,
)
from irksome import GaussLegendre, TimeStepper


def nl_solve_setup(F, t, dt, state, cfg, bcs=None, **solver_param_overrides):
    butcher_tableau = GaussLegendre(cfg["time"]["order"])
    solver_params = {
        "snes_max_it": 100,
        "snes_linesearch_type": "l2",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "mat_type": "aij",
        "pc_factor_mat_solver_type": "mumps",
    }
    if cfg["debug"]:
        solver_params["ksp_monitor"] = None
        solver_params["snes_monitor"] = None
    solver_params.update(solver_param_overrides)
    return TimeStepper(
        F, butcher_tableau, t, dt, state, bcs=bcs, solver_parameters=solver_params
    )


def _normalise(cfg):
    # Shorter references to various config sections for the sake of brevity
    constants = cfg["constants"]
    mesh = cfg["mesh"]
    model = cfg["model"]
    phys = cfg["physical"]

    # Normalisation factors
    # N.B. T norm is in 1/eV, phi norm is in C/eV
    norm = dict(
        charge=1 / constants["e"],
        n=1 / phys["n_0"],
        Ltrans=1 / phys["rho_s0"],
        Lpar=1 / phys["R"],
        T=1 / phys["T_e0"],
        phi=constants["e"] / phys["T_e0"],
        time=phys["c_s0"] / phys["R"],
    )
    # Derive mass normalisation from other quantities (convert phi norm from C/eV to C/J)
    norm["mass"] = (
        (norm["phi"] / constants["e"])
        * norm["time"]
        * norm["time"]
        / norm["Ltrans"]
        / norm["Ltrans"]
        * norm["charge"]
    )
    cfg["norm_factors"] = norm

    # Space norm
    mesh["Lz"] = mesh["Lz"] * norm["Lpar"]
    mesh["zmin"] = mesh["zmin"] * norm["Lpar"]
    for key in ["dx", "Lx", "Ly", "radius", "xmin", "ymin"]:
        if key in mesh:
            mesh[key] = mesh[key] * norm["Ltrans"]

    # Time norm
    time = cfg["time"]
    for key in ["t_start", "t_end"]:
        time[key] = time[key] * norm["time"]

    # Store some other normalised quantities for use in the ICs and BCs
    cfg["normalised"] = dict(
        c_s0=phys["c_s0"] * norm["Lpar"] / norm["time"],
        e=constants["e"] * norm["charge"],
        Ls=model["Ls"] * norm["Ltrans"],
        m_e=phys["m_e"] * norm["mass"],
        m_i=phys["m_i"] * norm["mass"],
        n_char=phys["n_char"] * norm["n"],
        n_init=model["n_init"] * norm["n"],
        omega_ci=phys["omega_ci"] / norm["time"],
        rs=model["rs"] * norm["Ltrans"],
        S0n=model["S0n"] * norm["n"] / norm["time"],
        S0T=model["S0T"] * norm["T"] / norm["time"],
        T_init=model["T_init"] * norm["T"],
    )
    cfg["normalised"]["B"] = phys["B"] * norm["mass"] / norm["time"] / norm["charge"]
    if mesh["type"] in ["circle", "rectangle"]:
        cfg["normalised"]["R"] = phys["R"] * norm["Ltrans"]
        # Dimensionless, but included here for consistency
        cfg["normalised"]["sigma"] = phys["sigma"] * norm["Ltrans"] / norm["Lpar"]
    else:
        cfg["normalised"]["dz"] = mesh["Lz"] / mesh["nz"]
        cfg["normalised"]["sigma_par"] = (
            phys["sigma_par"]
            * norm["time"]
            * norm["n"]
            * norm["charge"]
            * norm["charge"]
            / norm["mass"]
        )


def overrule_param_val(d, k, new_val, condition, msg):
    if condition:
        PETSc.Sys.Print(msg)
        d[k] = new_val


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
    Lphi = (
        -(grad(phi_tri)[0] * grad(phi_test)[0] + grad(phi_tri)[1] * grad(phi_test)[1])
        * dx
    )
    Rphi = Constant(rhs_fac) * w * phi_test * dx

    if bcs is None:
        # D0 on all boundaries
        bcs = DirichletBC(phi_space, 0, cfg["mesh"]["all_bdy_lbl"])

    phi_problem = LinearVariationalProblem(Lphi, Rphi, phi, bcs=bcs)
    solver_params = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    return LinearVariationalSolver(phi_problem, solver_parameters=solver_params)


def _process_params(cfg):
    """
    Set some default parameter values and add derived parameters
    """
    set_default_param(cfg, "output_base", "rogers-ricci")
    set_default_param(cfg, "debug", False)

    time_cfg = cfg["time"]
    # Time-related defaults
    set_default_param(time_cfg, "order", 1)
    set_default_param(time_cfg, "info_freq", 10)
    set_default_param(time_cfg, "output_freq", 10)
    set_default_param(time_cfg, "num_steps", 1000)

    # Add constants in SI-eV
    constants = dict(
        c=3e8, e=1.602e-19, kB=8.6173303e-5, m_e=9.1093837e-31, m_p=1.67e-27
    )
    cfg["constants"] = constants

    # Set model defaults
    model_cfg = cfg["model"]
    set_default_param(model_cfg, "is_isothermal", False)
    set_default_param(model_cfg, "coulomb_fac_enabled", False)
    set_default_param(model_cfg, "Ls_boost", 1.0)
    set_default_param(model_cfg, "elec_ion_mass_ratio", 400.0)
    set_default_param(model_cfg, "n_init", 1e18)
    set_default_param(model_cfg, "start_from_steady_state", True)
    set_default_param(model_cfg, "T_init", 6.0)

    # Set numerics defaults
    num_cfg = cfg["numerics"]
    set_default_param(num_cfg, "discretisation", "CG")
    set_default_param(num_cfg, "do_streamline_upwinding", True)
    assert cfg["numerics"]["discretisation"] in [
        "CG",
        "DG",
    ], "Invalid discretisation type was set"
    if cfg["numerics"]["do_streamline_upwinding"]:
        overrule_param_val(
            cfg["numerics"],
            "do_streamline_upwinding",
            False,
            cfg["numerics"]["discretisation"] == "DG",
            "Using DG: Ignoring do_streamline_upwinding=True",
        )

    # Set phys defaults
    phys_cfg = cfg["physical"]
    #  N.B. R is plasma column radius, *not* the transverse size of the domain!
    set_default_param(phys_cfg, "Lambda", 3.0)
    set_default_param(phys_cfg, "Lz", 18.0)
    set_default_param(phys_cfg, "m_i", 4 * constants["m_p"])
    # Unless defaults are overridden, use mass-boosted electrons as per paper; m_e = m_i/400 = m_p/100
    set_default_param(
        phys_cfg, "m_e", phys_cfg["m_i"] / model_cfg["elec_ion_mass_ratio"]
    )
    set_default_param(phys_cfg, "n_char", 2e18)
    set_default_param(phys_cfg, "n_0", 2e18)
    set_default_param(phys_cfg, "nu", 0.03)
    set_default_param(phys_cfg, "omega_ci", 9.6e5)
    set_default_param(phys_cfg, "R", 0.5)
    set_default_param(phys_cfg, "T_e0", 6.0)

    # Set mesh defaults
    mesh_cfg = cfg["mesh"]
    set_default_param(mesh_cfg, "type", "cuboid")
    mesh_type = mesh_cfg["type"]
    if mesh_type in ["cuboid", "rectangle"]:
        set_default_param(mesh_cfg, "nx", 1024)
        set_default_param(mesh_cfg, "ny", 1024)
        set_default_param(mesh_cfg, "use_hex", True)
    elif mesh_type == "circle":
        mesh_cfg["nx"] = 128
    elif mesh_type == "cylinder":
        mesh_cfg["longitudinal_axis"] = 2
        set_default_param(mesh_cfg, "ref_level", 3)
    else:
        raise ValueError(f"{mesh_type} is an invalid mesh type")

    if mesh_type not in ["circle", "rectangle"]:
        set_default_param(mesh_cfg, "nz", 64)

    # Derived physical params
    phys_cfg["B"] = phys_cfg["omega_ci"] * phys_cfg["m_i"] / constants["e"]
    phys_cfg["c_s0"] = math.sqrt(phys_cfg["T_e0"] * constants["e"] / phys_cfg["m_i"])
    phys_cfg["rho_s0"] = phys_cfg["c_s0"] / phys_cfg["omega_ci"]
    phys_cfg["c_s0_over_R"] = phys_cfg["c_s0"] / phys_cfg["R"] / phys_cfg["Lz"]
    phys_cfg["L"] = 100 * phys_cfg["rho_s0"]
    if mesh_type in ["circle", "rectangle"]:
        phys_cfg["sigma"] = 1.5 * phys_cfg["R"] / phys_cfg["Lz"]
    else:
        phys_cfg["sigma_par"] = (
            constants["e"]
            * constants["e"]
            * phys_cfg["n_0"]
            * phys_cfg["R"]
            / phys_cfg["m_i"]
            / phys_cfg["c_s0"]
            / phys_cfg["nu"]
        )

    # Derived mesh params
    mesh_cfg["Lz"] = phys_cfg["Lz"]
    mesh_cfg["zmin"] = -phys_cfg["Lz"] / 2
    if mesh_cfg["type"] in ["circle", "cuboid", "rectangle"]:
        mesh_cfg["dx"] = phys_cfg["L"] / mesh_cfg["nx"]
        mesh_cfg["Lx"] = phys_cfg["L"]
        mesh_cfg["Ly"] = phys_cfg["L"]
        mesh_cfg["xmin"] = -phys_cfg["L"] / 2
        mesh_cfg["ymin"] = -phys_cfg["L"] / 2
    elif mesh_type == "cylinder":
        mesh_cfg["ncells_tranverse"] = 2 ** (2 * mesh_cfg["ref_level"] + 3)
        mesh_cfg["radius"] = phys_cfg["L"]

    # Derived model params
    model_cfg["Ls"] = 0.5 * model_cfg["Ls_boost"] * phys_cfg["rho_s0"]
    model_cfg["rs"] = 20 * phys_cfg["rho_s0"]
    model_cfg["S0n"] = (
        model_cfg["S0n_fac"] * phys_cfg["n_0"] * phys_cfg["c_s0"] / phys_cfg["R"]
    )
    model_cfg["S0T"] = (
        model_cfg["S0T_fac"] * phys_cfg["T_e0"] * phys_cfg["c_s0"] / phys_cfg["R"]
    )

    # By default run between t=0 and t=12*R/c_s0
    set_default_param(time_cfg, "t_start", 0.0)
    set_default_param(time_cfg, "t_end", 12 * phys_cfg["R"] / phys_cfg["c_s0"])

    # # Check quantities in cgs match paper (not quite...)
    # print(f"c_s0 = {100*phys_cfg['c_s0']:.1E} cm/s")
    # print(f"rho_s0 = {100*phys_cfg['rho_s0']:.1E} cm")
    # print(f"c_s0_over_R = {phys_cfg['c_s0_over_R']:.1E} Hz")


def read_rr_config(fname):
    # Read Rogers & Ricci config file (expected next to this script)
    return read_yaml_config(
        fname, process_derived=_process_params, normalise=_normalise
    )


def rr_DG_upwind_term(tri, test, phi, mesh, cfg):
    vExB = rr_ExB_vel(phi, cfg)
    norms = FacetNormal(mesh)
    vExB_n = 0.5 * (dot(vExB, norms) + abs(dot(vExB, norms)))
    return (
        vExB_n("-") * (tri("-") - tri("+")) * test("-") * dS
        + vExB_n("+") * (tri("+") - tri("-")) * test("+") * dS
    )


def rr_ExB_vel(phi, cfg):
    one_over_B = Constant(1 / cfg["normalised"]["B"])
    return as_vector([-one_over_B * grad(phi)[1], one_over_B * grad(phi)[0]])


def rr_src_term(fspace, x, y, var, cfg):
    """
    Assemble a source term function on space [fspace] for variable [var],
    fetching corresponding scaling params from [cfg] and evaluating a
    tanh function over mesh coords [x],[y]
    """
    func = Function(fspace, name=f"{var}_src")
    func.interpolate(rr_src_ufl(x, y, var, cfg))
    return func


def rr_src_ufl(x, y, var, cfg):
    """
    Get UFL representing a source term for variable [var],
    fetching corresponding scaling params from [cfg] and evaluating a
    tanh function over mesh coords [x],[y]
    """
    r = sqrt(x * x + y * y)
    fac = cfg["normalised"][f"S0{var}"]
    Ls = cfg["normalised"]["Ls"]
    rs = cfg["normalised"]["rs"]
    return fac * (1 - tanh((r - rs) / Ls)) / 2


def rr_steady_state(x, y, cfg):
    cfg_norm = cfg["normalised"]

    # Fudge to get steady state to run - boost Ls in the ICs (but not the equations themselves)
    # Means we only have phi ~ 3T in the central blob, not outside
    norm_soft = dict(cfg_norm)
    norm_soft["Ls"] = norm_soft["Ls"] * 5

    sigma_cs_over_R = Constant(norm_soft["sigma"] * norm_soft["c_s0"] / norm_soft["R"])
    n_init = rr_src_ufl(x, y, "n", dict(normalised=norm_soft)) / sigma_cs_over_R
    T_init = rr_src_ufl(x, y, "T", dict(normalised=norm_soft)) / (
        sigma_cs_over_R * 2 / 3
    )

    r = sqrt(x * x + y * y)
    Ls = norm_soft["Ls"]
    rs = norm_soft["rs"]
    rn = (r - rs) / Ls
    rsoft = sqrt(x * x + y * y + 1e-2)
    w_init = (
        3
        * norm_soft[f"S0T"]
        / 2
        / (sigma_cs_over_R * 2 / 3)
        * ((2 * tanh(rn) / cosh(rn) ** 2) / Ls**2 - 1 / cosh(rn) ** 2 / rsoft / Ls)
    )
    # phi gets calculated before the first step, but it should be:
    phi_init = 3 * T_init
    return n_init, T_init, w_init


def rr_SU_term(tri, test, phi, h, cfg, vel_par=None, eps=1e-2):
    vel = rr_ExB_vel(phi, cfg)
    if vel_par is not None:
        vel = as_vector([vel[0], vel[1], vel_par])
    return (
        0.5
        * h
        * (dot(vel, grad(tri)))
        * dot(vel, grad(test))
        * (1 / sqrt((vel[0]) ** 2 + (vel[1]) ** 2 + eps * eps))
        * dx
    )
