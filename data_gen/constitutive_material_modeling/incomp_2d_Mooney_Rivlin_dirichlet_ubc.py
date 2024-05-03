## Author: Minglang Yin, minglang_yin@brown.edu
## Uniaxial pulling of a 3D plate
## Holzapfel-Gasser-Odgen Model
## Clamped bottom
## Impose displacement BC on the top

from dolfin import *
import random
# from mshr import *
import os
import numpy as np
from numpy.random import random
import sys
from fem import gaussian_random_field
import math
import time

set_number = int(sys.argv[1])
print(set_number)
np.random.seed(set_number)
# make new directory
path = 'ubctest_final_order2_0075_0075_coef5'
os.makedirs(path, exist_ok=True)

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

# Define the solver parameters
solver_parameters = {"newton_solver": {"maximum_iterations": 50,
                                       "relative_tolerance": 1e-8,
                                       "absolute_tolerance": 1e-9}}
# Create mesh and define function space
n = 40
order = 2
mesh = RectangleMesh(Point(0, 0), Point(1.0, 1.0), n, n)
# mesh = RectangleMesh.create([Point(0,0),Point(1.0,1.0)],[20,20],CellType.Type.quadrilateral)
V1 = VectorElement("CG", mesh.ufl_cell(), 2)  # displacement finite element
V2 = FiniteElement("CG", mesh.ufl_cell(), 1)  # lagrange multiplier
V = FunctionSpace(mesh, MixedElement([V1, V2]))
Vb = FunctionSpace(mesh, V1)
# V = VectorFunctionSpace(mesh, "CG", 1)
# W = TensorFunctionSpace(mesh, "CG", 1)
d = mesh.geometry().dim()


def getNodeCoordinates(V, nFields):
    dofxFlat = V.tabulate_dof_coordinates()
    dofx = dofxFlat.reshape((-1, d))
    # print(dofx)
    nNodes = dofx.shape[0] // nFields
    nodex = np.zeros((nNodes, d))
    for i in range(0, nNodes):
        nodex[i, :] = dofx[nFields * i, :]
    return nodex, nNodes


Nodex, nNodes = getNodeCoordinates(V, 2)

# Mark boundary subdomians
bot = CompiledSubDomain("near(x[1], side) && on_boundary", side=0.0)
top = CompiledSubDomain("near(x[1], side) && on_boundary", side=1.0)
left = CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side=1.0)


def top_bd(x, on_boundary):
    return x[1] > 1 - DOLFIN_EPS and on_boundary


# Define Dirichlet boundary (x = 0 or x = 1)
top_disp_1 = Expression("0.0", degree=1)
top_disp_2 = Expression("0.0", degree=1)
bot_disp_1 = Expression("0.0", degree=1)
bot_disp_2 = Expression("0.0", degree=1)
# top_disp = Expression(("t*0.0","t*"+str(uy)), t = 1, degree=1)
left_disp_1 = Expression("0.0", degree=1)
left_disp_2 = Expression("0.0", degree=1)
right_disp_1 = Expression("0.0", degree=1)
right_disp_2 = Expression("0.0", degree=1)
# right_disp = Expression(("t*"+str(ux),"t*0.0"), t = 1, degree=1)

# bc1 = DirichletBC(V.sub(0), Constant((0.,0.)), top)
top_disp = Expression(("-0.5*x[1]", "0.5*x[0]"), degree=1)
bc1 = DirichletBC(V.sub(0), top_disp, top)
# bc2 = DirichletBC(V.sub(1), top_disp_2, top)
# bc3 = DirichletBC(V.sub(0), Constant((0.,0.)), bot)
bot_disp = Expression(("-0.5*x[1]", "0.5*x[0]"), degree=1)
bc3 = DirichletBC(V.sub(0), bot_disp, bot)
# bc4 = DirichletBC(V.sub(1), bot_disp_2, bot)
# bc5 = DirichletBC(V.sub(0), Constant((0.,0.)), left)
left_disp = Expression(("-0.5*x[1]", "0.5*x[0]"), degree=1)
bc5 = DirichletBC(V.sub(0), left_disp, left)
# bc6 = DirichletBC(V.sub(1), left_disp_2, left)
# bc7 = DirichletBC(V.sub(0), Constant((0.,0.)), right)
right_disp = Expression(("-0.5*x[1]", "0.5*x[0]"), degree=1)
bc7 = DirichletBC(V.sub(0), right_disp, right)
# bc8 = DirichletBC(V.sub(1), right_disp_2, right)
# bcs = [bc1, bc2, bc3, bc4, bc5, bc6, bc7, bc8]
bcs = [bc1, bc3, bc5, bc7]
# bcs = [bc1, bc3]

# Define functions
du = TrialFunction(V)  # Incremental displacement
v = TestFunction(V)  # Test function
u_t = Function(V)  # Displacement from previous iteration
(u, p) = split(u_t)
(v_u, v_p) = split(v)
# T  = Function(V)                 # Displacement from previous iteration
B = Function(Vb)  # Constant((0.0, 0.0))  # Body force per unit volume
# (B, tt) = split(B_t)

# Kinematics
d = 2  # len(u)
I = Identity(d)  # Identity tensor
grad_u = grad(u)
F = I + grad_u
eps = 0.5 * (grad_u + grad_u.T)  # small def strain
ev = eps[0, 0] + eps[1, 1]
C = F.T * F
CB = F * F.T
I1 = tr(C)
I2 = 0.5 * (I1 * I1 - tr(C * C))
# sig = 0.38 * eps - p * I
A10 = 0.075
A01 = 0.075
sig = 2 * A10 * CB - 2 * A01 * inv(CB) - p * I


# psi = 0.5*inner(sig, grad_u)

# Solve variational problem
# B_array = np.zeros((2*nNodes))

class MyExpression0(UserExpression):
    def eval(self, value, x):
        value[0] = 0.0
        value[1] = 0.0
        # c=5.0
        c1 = 1.0
        # r=np.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2)
        # if np.abs(x[0]-0.5)<0.2+1e-5 and np.abs(x[1]-0.5)<0.2+1e-5:
        #     for k1 in range(1, 1 + order):
        #         for k2 in range(1, 1 + order):
        #             value[0] += exp(-k1 * k2 * 0.1) * coef_a[k1 - 1, k1 - 1] * sin(k1 * 3.1415925 * (x[0]-0.3)/0.4) * sin(k2 * 3.1415925 * (x[1]-0.3)/0.4)
        #             value[1] += exp(-k1 * k2 * 0.1) * coef_b[k1 - 1, k2 - 1] * sin(k1 * 3.1415925 * (x[0]-0.3)/0.4) * sin(k2 * 3.1415925 * (x[1]-0.3)/0.4)
        for k1 in range(1, 1 + order):
            for k2 in range(1, 1 + order):
                value[0] += c1 * exp(-k1 * k2 * 0.1) * coef_a[k1 - 1, k1 - 1] * sin(k1 * 3.1415925 * x[0]) * sin(
                    k2 * 3.1415925 * x[1])
                value[1] += c1 * exp(-k1 * k2 * 0.1) * coef_b[k1 - 1, k2 - 1] * sin(k1 * 3.1415925 * x[0]) * sin(
                    k2 * 3.1415925 * x[1])

    #        value[0] = value[0] - c * x[1]
    #        value[1] = value[1] + c * x[0]

    def value_shape(self):
        return (2,)


for i in range(set_number, set_number + 1):
    # Generate random body force
    # B1_random = 0.08 * gaussian_random_field(alpha=5.0, size=41, flag_normalize=True)
    # B2_random = 0.08 * gaussian_random_field(alpha=5.0, size=41, flag_normalize=True)

    for j in range(nNodes):
        x_coord = Nodex[j, 0]
        y_coord = Nodex[j, 1]
        x_ind = math.floor((x_coord + 1e-10) / 0.05)
        y_ind = math.floor((y_coord + 1e-10) / 0.05)
        # B_array[2*j] = B1_random[y_ind,x_ind]
        # B_array[2*j+1] = B2_random[y_ind,x_ind]
        # print(j, x_coord, y_coord, T_random[y_ind,x_ind])
    # B.vector().set_local(B_array)
    # B = Expression(("0.0", "a*sin(3.1415925*x[0])*sin(3.1415925*x[1])"), a=0.1, degree=2)
    coef_a = (random(size=[order, order]) - 0.5) * 5
    coef_b = (random(size=[order, order]) - 0.5) * 5
    print(coef_a)
    print(coef_b)
    B = MyExpression0()

    Func = inner(grad(v_u), sig) * dx - dot(v_u, B) * dx + v_p * ev * dx

    solve(Func == 0, u_t, bcs, form_compiler_parameters=ffc_options, solver_parameters=solver_parameters)
    (u, p) = u_t.split()

    F = I + grad(u)  # Deformation gradient
    b = F * F.T
    # strch = np.linalg.eigvals(b)
    # print(strch)
    file_name_1 = path + "/coords_" + str(i) + ".txt"
    file_name_2 = path + "/displacement_" + str(i) + ".txt"
    file_name_3 = path + "/bodyforce_" + str(i) + ".txt"

    f_x = open(file_name_1, 'w')
    f_u = open(file_name_2, 'w')
    f_B = open(file_name_3, 'w')
    for xi in range(0, n + 1):
        for xj in range(0, n + 1):
            x_cords = xi * 1.0 / n
            y_cords = xj * 1.0 / n
            f_x.write("%f %f\n" % (x_cords, y_cords))
            f_u.write("%f %f\n" % (u([x_cords, y_cords])[0], u([x_cords, y_cords])[1]))
            # f_B.write("%f %f\n" % (B([x_cords, y_cords])[0], B([x_cords, y_cords])[1]))
            f_B.write("%f %f\n" % (B([x_cords, y_cords])[0], B([x_cords, y_cords])[1]))

    # f_B.write("\n")
    # f_u.write("\n")
    # f_x.write("\n")
    f_B.close()
    f_u.close()
    f_x.close()

# np.savetxt("stress_strain.dat", np.array(u_his))
