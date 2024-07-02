# LAPD-like_CG_v2.py
# Attempt at time-dependent solver of LAPD-like equations, with upwind flux
# based upon SOL_3D_DG_upwind_tdep_irksome_dev.py (longitudinal dynamics)
# and Nektar-Driftwave_port_irksome_STFC_v2_working_submit.py (transverse dynamics)
# this version fixes wrong equation for longitudial velocity cpt

from firedrake import *
import math
from irksome import Dt, GaussLegendre, MeshConstant, TimeStepper
import time

meshres = 32
h=1.0/meshres
mesh = BoxMesh(16, meshres, meshres, 2.0, 0.2, 0.2, hexahedral=True)

V1 = FunctionSpace(mesh, "CG", 1) # n
V2 = FunctionSpace(mesh, "CG", 1)  # u - velocity x-cpt
V3 = FunctionSpace(mesh, "CG", 1) # w
V = V1*V2*V3
V4 = FunctionSpace(mesh, "CG", 1) # phi

# time parameters (4.0 / 100 is the standard for longitudinal-only)
T = 8.0
timeres = 400
t = Constant(0.0)
dt = Constant(T/timeres)

# parameters for irksome
butcher_tableau = GaussLegendre(1)
#butcher_tableau = GaussLegendre(2)  # bit slow on my laptop, makes it take 8 hrs on 8 cores

# model parameters
nstar = Constant(1.0)
Temp = 1.0
width_T = 0.05  # transverse width for Gaussian source

x = SpatialCoordinate(mesh)
nuw = Function(V)
n, u, w = split(nuw)
v1, v2, v3 = TestFunctions(V)
phi = TrialFunction(V4)
v4 = TestFunction(V4)
phi_s = Function(V4)

# sonic outflow equilibrium init data
#nuw.sub(0).interpolate((nstar/sqrt(Temp))*(1+sqrt(1-x[0]*x[0])))
#nuw.sub(1).interpolate((sqrt(Temp)/(x[0]))*(1-sqrt(1-x[0]*x[0])))

# source function, amplitude was simple cranked up until nontrivial behaviour was seen
nstarFunc = Function(V1)
nstarFunc.interpolate(nstar*0.0 + 100.0*exp(-((x[1]-0.1)**2+(x[2]-0.1)**2)/(2*width_T**2)))

# TRIALCODE check init data
File("LAPD-like_CG_v2_init.pvd").write(nuw.sub(0), nuw.sub(1), nuw.sub(2))
#quit()

# dev version with longitudinal and transverse dynamics
# last 3 terms are the artificial viscosity
F = ((Dt(n)*v1)*dx + (n*Dt(u)*v2 + Dt(w)*v3)*dx) \
   + (v1*div(n*as_vector([u,grad(phi_s)[2],-grad(phi_s)[1]]))-v1*nstarFunc)*dx \
   + (v3*div(w*as_vector([u,grad(phi_s)[2],-grad(phi_s)[1]]))+v3*grad(n*u)[0])*dx \
   + (nstarFunc*u*v2+v2*n*inner(grad(u),as_vector([u,grad(phi_s)[2],-grad(phi_s)[1]]))+Temp*grad(n)[0]*v2)*dx \
   + 0.3*(0.5*h*(dot(as_vector([u,grad(phi_s)[2],-grad(phi_s)[1]]),grad(n)-as_vector([0,0,nstarFunc])))*dot(as_vector([u,grad(phi_s)[2],-grad(phi_s)[1]]),grad(v1)))*(1/sqrt((grad(phi_s)[1])**2+(grad(phi_s)[2])**2+u**2+0.0001))*dx \
   + 0.3*(0.5*h*(dot(as_vector([u,grad(phi_s)[2],-grad(phi_s)[1]]),grad(w)))*dot(as_vector([u,grad(phi_s)[2],-grad(phi_s)[1]]),grad(v3)))*(1/sqrt((grad(phi_s)[1])**2+(grad(phi_s)[2])**2+u**2+0.0001))*dx \
   + 0.3*(0.5*h*(dot(as_vector([u,grad(phi_s)[2],-grad(phi_s)[1]]),grad(n*u)))*dot(as_vector([u,grad(phi_s)[2],-grad(phi_s)[1]]),grad(v2)))*(1/sqrt((grad(phi_s)[1])**2+(grad(phi_s)[2])**2+u**2+0.0001))*dx \

# params taken from Cahn-Hilliard example cited above
params = {'snes_monitor': None, 'snes_max_it': 100,
          'snes_linesearch_type': 'l2',
          'ksp_type': 'preonly',
          'pc_type': 'lu', 'mat_type': 'aij',
          'pc_factor_mat_solver_type': 'mumps'}

# Dirichlet BCs are needed for boundary velocity

bc_outflow_1 = DirichletBC(V.sub(1),-1,1)
bc_outflow_2 = DirichletBC(V.sub(1), 1,2)

bc_n = DirichletBC(V.sub(0), 0.0, [3,4,5,6])

stepper = TimeStepper(F, butcher_tableau, t, dt, nuw, solver_parameters=params, bcs=[bc_outflow_1, bc_outflow_2])

# elliptic solve for potential
Lphi = inner(grad(phi),grad(v4))*dx
Rphi = -w*v4*dx
#phi_s = Function(V4)
bc1 = DirichletBC(V4, 0, 'on_boundary')

# this is intended to be direct solver - but now changed to GMRES
linparams = {"mat_type": "aij",
          "snes_type": "ksponly",
          "ksp_type": "gmres",
          "pc_type": "lu", 'mat_type': 'aij',
          'pc_factor_mat_solver_type': 'mumps'}

nullspace = VectorSpaceBasis(constant=True)

# end of stuff for elliptic solve

outfile = VTKFile("LAPD-like_CG_v2.pvd")

start = time.time()

while float(t) < float(T):
    if (float(t) + float(dt)) >= T:
        dt.assign(T - float(t))

    solve(Lphi==Rphi, phi_s, nullspace=nullspace, solver_parameters=linparams, bcs=bc1)

    nuw.sub(0).rename("density")
    nuw.sub(1).rename("velocity")
    nuw.sub(2).rename("vorticity")
    p = Function(V1)
    p.interpolate(nuw.sub(0)*nuw.sub(1))
    p.rename("momentum density")
    phi_s.rename("potential")
    outfile.write(nuw.sub(0), nuw.sub(1), nuw.sub(2), phi_s, p)


    stepper.advance()
    t.assign(float(t) + float(dt))
    print(float(t), float(dt))

end = time.time()
wall_time = end-start

print("done.")
print("\n")
print("wall time:"+str(wall_time)+"\n")
