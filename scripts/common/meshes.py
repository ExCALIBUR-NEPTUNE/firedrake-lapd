from firedrake import ExtrudedMesh, mesh, TensorBoxMesh, ufl
import numpy as np
from pyop2.mpi import COMM_WORLD
from firedrake import PETSc
from firedrake.cython import dmcommon


def set_up_mesh(opts):
    try:
        mesh_opts = opts["mesh"]
        mesh_type = mesh_opts["type"]
        if mesh_type == "cuboid":
            mesh = BoxMesh(
                mesh_opts["nx"],
                mesh_opts["ny"],
                mesh_opts["nz"],
                mesh_opts["Lx"],
                mesh_opts["Ly"],
                mesh_opts["Lz"],
                lower=(mesh_opts["xmin"], mesh_opts["ymin"], mesh_opts["zmin"]),
                hexahedral=mesh_opts["use_hex"],
            )
        elif mesh_type == "cylinder":
            ref_level = mesh_opts["ref_level"]
            # Store num cells in transverse slice
            mesh_opts["ncells_tranverse"] = 2 ** (2 * ref_level + 3)
            mesh = CylinderMesh(
                mesh_opts["radius"],
                mesh_opts["nz"],
                mesh_opts["Lz"],
                longitudinal_axis=mesh_opts["longitudinal_axis"],
                refinement_level=ref_level,
            )
        else:
            raise ValueError(f"mesh_type [{mesh_type}] not recognised")
    except KeyError:
        PETSc.Sys.Print("Unset parameter encountered in set_up_mesh()")
        raise

    return mesh


def BoxMesh(
    nx,
    ny,
    nz,
    Lx,
    Ly,
    Lz,
    lower=(0.0, 0.0, 0.0),
    hexahedral=False,
    reorder=None,
    distribution_parameters=None,
    diagonal="default",
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of a 3D box.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg Lz: The extent in the z direction
    :kwarg lower: lower-left corner of box (min vals)
    :kwarg hexahedral: (optional), creates hexahedral mesh.
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg diagonal: Two ways of cutting hexadra, should be cut into 6
        tetrahedra (``"default"``), or 5 tetrahedra thus less biased
        (``"crossed"``)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on.

    The boundary surfaces are numbered as follows:

    * 1: plane x == lower[0]
    * 2: plane x == lower[0] + Lx
    * 3: plane y == lower[1]
    * 4: plane y == lower[1] + Ly
    * 5: plane z == lower[2]
    * 6: plane z == lower[2] + Lz
    """
    for n in (nx, ny, nz):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")
    if hexahedral:
        sizes = (Lx, Ly, Lz)
        upper = tuple([si + li for li, si in zip(lower, sizes)])
        plex = PETSc.DMPlex().createBoxMesh(
            (nx, ny, nz),
            lower=lower,
            upper=upper,
            simplex=False,
            periodic=False,
            interpolate=True,
            comm=comm,
        )
        plex.removeLabel(dmcommon.FACE_SETS_LABEL)
        nvert = 4  # num. vertices on faect

        # Apply boundary IDs
        plex.createLabel(dmcommon.FACE_SETS_LABEL)
        plex.markBoundaryFaces("boundary_faces")
        coords = plex.getCoordinates()
        coord_sec = plex.getCoordinateSection()
        cdim = plex.getCoordinateDim()
        assert cdim == 3
        if plex.getStratumSize("boundary_faces", 1) > 0:
            boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
            xtol = Lx / (2 * nx)
            ytol = Ly / (2 * ny)
            ztol = Lz / (2 * nz)
            for face in boundary_faces:
                face_coords = plex.vecGetClosure(coord_sec, coords, face)
                if all([abs(face_coords[0 + cdim * i]- lower[0])  < xtol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
                if all(
                    [abs(face_coords[0 + cdim * i] - upper[0]) < xtol for i in range(nvert)]
                ):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
                if all([abs(face_coords[1 + cdim * i] - lower[1]) < ytol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
                if all(
                    [abs(face_coords[1 + cdim * i] - upper[1]) < ytol for i in range(nvert)]
                ):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 4)
                if all([abs(face_coords[2 + cdim * i]- lower[2]) < ztol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 5)
                if all(
                    [abs(face_coords[2 + cdim * i] - upper[2]) < ztol for i in range(nvert)]
                ):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 6)
        plex.removeLabel("boundary_faces")
        m = mesh.Mesh(
            plex,
            reorder=reorder,
            distribution_parameters=distribution_parameters,
            name=name,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
            comm=comm,
        )
        return m
    else:
        xcoords = np.linspace(lower[0], lower[0] + Lx, nx + 1, dtype=np.double)
        ycoords = np.linspace(lower[1], lower[1] + Ly, ny + 1, dtype=np.double)
        zcoords = np.linspace(lower[2], lower[2] + Lz, nz + 1, dtype=np.double)
        return TensorBoxMesh(
            xcoords,
            ycoords,
            zcoords,
            reorder=reorder,
            distribution_parameters=distribution_parameters,
            diagonal=diagonal,
            comm=comm,
            name=name,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
        )


def _DiskMesh(
    radius,
    refinement_level=0,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of the unit disk in 2D

    :kwarg refinement_level: optional number of refinements (0 is a diamond)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    vertices = np.array(
        [
            [0, 0],
            [radius, 0],
            [radius, radius],
            [0, radius],
            [-radius, radius],
            [-radius, 0],
            [-radius, -radius],
            [0, -radius],
            [radius, -radius],
        ],
        dtype=np.double,
    )

    cells = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 7],
            [0, 7, 8],
            [0, 8, 1],
        ],
        np.int32,
    )

    plex = mesh.plex_from_cell_list(
        2, cells, vertices, comm, mesh._generate_default_mesh_topology_name(name)
    )

    # mark boundary facets
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        for face in boundary_faces:
            plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
    plex.removeLabel("boundary_faces")
    plex.setRefinementUniform(True)
    for i in range(refinement_level):
        plex = plex.refine()

    coords = plex.getCoordinatesLocal().array.reshape(-1, 2)
    for x in coords:
        norm = np.sqrt(np.dot(x, x))
        if norm > 1.0 / (1 << (refinement_level + 1)):
            t = np.max(np.abs(x)) / norm
            x[:] *= t

    m = mesh.Mesh(
        plex,
        dim=2,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    return m


def CylinderMesh(
    radius, nh, dh, refinement_level=3, longitudinal_axis=2, comm=COMM_WORLD
):
    disk = _DiskMesh(radius, refinement_level=refinement_level, comm=comm)
    cyl = ExtrudedMesh(disk, nh, layer_height=dh / nh, extrusion_type="uniform")

    # Longitudinal axis is 2 after extrusion. Swap coords if necessary

    if longitudinal_axis == 0:
        swap = np.array(cyl.coordinates.dat.data[:, 2])
        cyl.coordinates.dat.data[:, 2] = np.array(cyl.coordinates.dat.data[:, 0])
        cyl.coordinates.dat.data[:, 0] = swap
    elif longitudinal_axis == 1:
        swap = np.array(cyl.coordinates.dat.data[:, 2])
        cyl.coordinates.dat.data[:, 2] = np.array(cyl.coordinates.dat.data[:, 1])
        cyl.coordinates.dat.data[:, 1] = swap
    elif longitudinal_axis == 2:
        pass  # no action required
    else:
        raise ValueError("longitudinal_axis must be 0, 1 or 2")

    return cyl
