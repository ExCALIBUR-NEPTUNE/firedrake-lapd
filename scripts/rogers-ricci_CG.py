from common import read_yaml_config, set_default_param, set_up_mesh
from firedrake import (
    Function,
    FunctionSpace,
    SpatialCoordinate,
    sqrt,
    tanh,
    TestFunction,
    TrialFunction,
    VTKFile,
)
import math


def process_params(cfg):
    """
    Set some default parameter values and add derived parameters
    """
    # Add constants in SI-eV
    constants = dict(e=1.602e-19, kB=8.6173303e-5, m_e=9.1093837e-31, m_p=1.67e-27)
    cfg["constants"] = constants

    # Set phys defaults
    phys_cfg = cfg["physical"]
    #  N.B. R is plasma column radius, *not* the transverse size of the domain!
    set_default_param(phys_cfg, "R", 0.5)
    set_default_param(phys_cfg, "Lz", 18.0)
    # Paper claims m_i = 400 m_e, but can only match rho_s0 (and therefore domain size) with a value of ~3*m_p ...
    set_default_param(phys_cfg, "m_i", 3 * constants["m_p"])
    set_default_param(phys_cfg, "n_0", 2e18)
    set_default_param(phys_cfg, "omega_ci", 9.6e5)

    # Set mesh defaults
    mesh_cfg = cfg["mesh"]
    set_default_param(mesh_cfg, "type", "cuboid")
    mesh_type = mesh_cfg["type"]
    if mesh_type == "cuboid":
        set_default_param(mesh_cfg, "nx", 1024)
        set_default_param(mesh_cfg, "ny", 1024)
        set_default_param(mesh_cfg, "use_hex", True)
    elif mesh_type == "cylinder":
        set_default_param(mesh_cfg, "longitudinal_axis", 2)
        set_default_param(mesh_cfg, "ref_level", 3)
    else:
        raise ValueError(f"{mesh_type} is an invalid mesh type")
    set_default_param(mesh_cfg, "nz", 64)

    # Derived physical params
    phys_cfg["c_s0"] = math.sqrt(phys_cfg["T_e0"] * constants["e"] / phys_cfg["m_i"])
    phys_cfg["rho_s0"] = phys_cfg["c_s0"] / phys_cfg["omega_ci"]
    phys_cfg["c_s0_over_R"] = phys_cfg["c_s0"] / phys_cfg["R"]
    phys_cfg["L"] = 100 * phys_cfg["rho_s0"]

    # Derived mesh params (could add normalisation here)
    mesh_cfg["Lz"] = phys_cfg["Lz"]
    mesh_cfg["zmin"] = -phys_cfg["Lz"] / 2
    if mesh_type == "cuboid":
        mesh_cfg["Lx"] = phys_cfg["L"]
        mesh_cfg["Ly"] = phys_cfg["L"]
        mesh_cfg["xmin"] = -phys_cfg["L"] / 2
        mesh_cfg["ymin"] = -phys_cfg["L"] / 2
    elif mesh_type == "cylinder":
        mesh_cfg["ncells_tranverse"] = 2 ** (2 * mesh_cfg["ref_level"] + 3)
        mesh_cfg["radius"] = phys_cfg["L"]

    # Derived model params
    model_cfg = cfg["model"]
    model_cfg["Ls"] = 0.5 * phys_cfg["rho_s0"]
    model_cfg["rs"] = 20 * phys_cfg["rho_s0"]
    model_cfg["S0n"] = (
        model_cfg["S0n_fac"] * phys_cfg["n_0"] * phys_cfg["c_s0"] / phys_cfg["R"]
    )
    model_cfg["S0T"] = (
        model_cfg["S0T_fac"] * phys_cfg["T_e0"] * phys_cfg["c_s0"] / phys_cfg["R"]
    )

    # # Check quantities in cgs match paper (not quite...)
    # print(f"c_s0 = {100*phys_cfg['c_s0']:.1E} cm/s")
    # print(f"rho_s0 = {100*phys_cfg['rho_s0']:.1E} cm")
    # print(f"c_s0_over_R = {phys_cfg['c_s0_over_R']:.1E} Hz")


def src_term(fspace, x, y, var, opts):
    """
    Assemble a source term function on space [fspace] for variable [var],
    fetching corresponding scaling params from [opts] and evaluating a
    tanh function over mesh coords [x],[y]
    """
    r = sqrt(x * x + y * y)
    fac = opts["model"][f"S0{var}"]
    Ls = opts["model"]["Ls"]
    rs = opts["model"]["rs"]
    func = Function(fspace)
    func.interpolate(fac * (1 - tanh((r - rs) / Ls)) / 2)
    return func


def rogers_ricci():
    # Read config file (expected next to this script)

    opts = read_yaml_config("rogers-ricci_config.yml", process_derived=process_params)
    # Generate mesh
    mesh = set_up_mesh(opts)
    x, y, z = SpatialCoordinate(mesh)

    # Function spaces
    V1 = FunctionSpace(mesh, "CG", 1)  # n
    V2 = FunctionSpace(mesh, "CG", 1)  # ui - parallel ion velocity
    V3 = FunctionSpace(mesh, "CG", 1)  # ue - parallel electron velocity
    V4 = FunctionSpace(mesh, "CG", 1)  # T
    V5 = FunctionSpace(mesh, "CG", 1)  # w
    V6 = FunctionSpace(mesh, "CG", 1)  # phi

    # Source functions
    nsrc = src_term(V1, x, y, "n", opts)
    Tsrc = src_term(V4, x, y, "T", opts)

    # Check the source functions look ok
    # outfile = VTKFile(f"src_funcs.pvd")
    # outfile.write(nsrc, Tsrc)


if __name__ == "__main__":
    rogers_ricci()
