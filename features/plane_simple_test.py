# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:09:44 2023

@author: devoi
"""
import pickle
import sympy as sp
import sympy.physics.mechanics as me
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from pydy.codegen.cython_code import CythonMatrixGenerator

import force_models as fm

plt.style.use(['science','ieee']) 
me.init_vprinting(use_latex='mathjax')

def heaviside_regularized(x, a=1000):
    return 0.5*sp.tanh(a*x) + 0.5
    
def form_generalized_speeds_vector(t,q):
    n = q.shape[0]
    u = sp.Matrix([me.dynamicsymbols(f"U_{i}") for i in range(1,n+1)])
    repl = {qi.diff(t) : ui for qi, ui in zip(q, u)}
    return u, repl

tic = time.perf_counter()
    
L1,L2,ax,ay,az,bx,by,bz,g = sp.symbols("L1 L2 a_x a_y a_z b_x b_y b_z g", 
                                       positive=True)
alpha_1,alpha_2 = sp.symbols("alpha_1 alpha_2", 
                             real=True)
Fu_NLG,Fv_NLG,Ft_NLG = sp.symbols("F_{uNLG} F_{vNLG} F_{tNLG}", 
                                  real=True)
Fu_LMLG,Fv_LMLG,Ft_LMLG = sp.symbols("F_{uLMLG} F_{vLMLG} F_{tLMLG}", 
                                     real=True)
Fu_RMLG,Fv_RMLG,Ft_RMLG = sp.symbols("F_{uRMLG} F_{vRMLG} F_{tRMLG}", 
                                     real=True)
M,Jxx,Jxz,Jzz,m1,m2 = sp.symbols("M J_x J_{xz} J_z m1 m2", 
                                 positive=True)
y,theta_x,theta_z,u1,v1,u2,v2,u3,v3 = me.dynamicsymbols("y theta_x theta_z u1 v1 u2 v2 u3 v3")
t = me.dynamicsymbols._t

inertial_frame = me.ReferenceFrame("I")
plane_frame = me.ReferenceFrame("P")
nlg_frame = me.ReferenceFrame("N")
lmlg_frame = me.ReferenceFrame("LM")
rmlg_frame = me.ReferenceFrame("RM")

plane_frame.orient(inertial_frame, 'Body', (theta_x, 0, theta_z), 'XYZ')
nlg_frame.orient( plane_frame, 'Body', (0, 0, alpha_1), 'XYZ')
lmlg_frame.orient(plane_frame, 'Body', (0, 0, alpha_2), 'XYZ')
rmlg_frame.orient(plane_frame, 'Body', (0, 0, alpha_2), 'XYZ')

origin = me.Point("O")
plane_CG = me.Point("CG")
plane_nlg_mount = me.Point("NLG")
plane_lmlg_mount = me.Point("LMLG")
plane_rmlg_mount = me.Point("RMLG")
nlg_axis = me.Point("NLGW")
lmlg_axis = me.Point("LMLGW")
rmlg_axis = me.Point("RMLGW")

plane_CG.set_pos(origin, y * inertial_frame.y)
plane_nlg_mount.set_pos(plane_CG, (-ax * plane_frame.x - 
                                    ay * plane_frame.y + 
                                    az * plane_frame.z))
plane_lmlg_mount.set_pos(plane_CG, (bx * plane_frame.x - 
                                    by * plane_frame.y + 
                                    bz * plane_frame.z))
plane_rmlg_mount.set_pos(plane_CG, (bx * plane_frame.x - 
                                    by * plane_frame.y - 
                                    bz * plane_frame.z))
nlg_axis.set_pos(plane_nlg_mount,   ((-L1 + u1) * nlg_frame.y + 
                                             v1 * nlg_frame.x))
lmlg_axis.set_pos(plane_lmlg_mount, ((-L2 + u2) * lmlg_frame.y + 
                                             v2 * lmlg_frame.x))
rmlg_axis.set_pos(plane_rmlg_mount, ((-L2 + u3) * rmlg_frame.y + 
                                             v3 * rmlg_frame.x))

origin.set_vel(inertial_frame, 0)
plane_CG.set_vel(inertial_frame, plane_CG.pos_from(origin).dt(inertial_frame))

plane_nlg_mount.v2pt_theory( plane_CG, inertial_frame, plane_frame)
plane_lmlg_mount.v2pt_theory(plane_CG, inertial_frame, plane_frame)
plane_rmlg_mount.v2pt_theory(plane_CG, inertial_frame, plane_frame)

nlg_axis.set_vel(inertial_frame, 
                 nlg_axis.pos_from(plane_nlg_mount).dt(inertial_frame))
lmlg_axis.set_vel(inertial_frame, 
                  lmlg_axis.pos_from(plane_lmlg_mount).dt(inertial_frame))
rmlg_axis.set_vel(inertial_frame, 
                  rmlg_axis.pos_from(plane_rmlg_mount).dt(inertial_frame))

print("System kinematics succesfully initialized.")

plane_inertia = (Jxx*me.outer(plane_frame.x, plane_frame.x) - 
                 Jxz*me.outer(plane_frame.x, plane_frame.z) - 
                 Jxz*me.outer(plane_frame.z, plane_frame.x) + 
                 Jzz*me.outer(plane_frame.z, plane_frame.z))

plane_body = me.RigidBody("Plane", 
                          plane_CG, 
                          plane_frame, 
                          M, 
                          (plane_inertia, plane_CG))
nlg_wheels = me.Particle("NLG Wheels", nlg_axis, m1)
lmlg_wheels = me.Particle("LMLG Wheels", lmlg_axis, m2)
rmlg_wheels = me.Particle("RMLG Wheels", rmlg_axis, m2)

T_CG = plane_body.kinetic_energy(inertial_frame)#.trigsimp().expand()
print("Kinetic energy of plane is computed.")
T_NLG = nlg_wheels.kinetic_energy(inertial_frame)#.trigsimp().expand()
print("Kinetic energy of NLG is computed.")
T_LMLG = lmlg_wheels.kinetic_energy(inertial_frame)#.trigsimp().expand()
print("Kinetic energy of left MLG is computed.")
T_RMLG = rmlg_wheels.kinetic_energy(inertial_frame)#.trigsimp().expand()
print("Kinetic energy of right MLG is computed.")

T = T_CG + T_NLG + T_LMLG + T_RMLG

G  = - M*g*inertial_frame.y
g1 = - m1*g*inertial_frame.y
g2 = - m2*g*inertial_frame.y

R_plane = (Fu_NLG*nlg_frame.y + Fu_LMLG*lmlg_frame.y + Fu_RMLG*rmlg_frame.y + 
           Fv_NLG*nlg_frame.x + Fv_LMLG*lmlg_frame.x + Fv_RMLG*rmlg_frame.x + G)
T_plane = (me.cross(plane_nlg_mount.pos_from(origin),  
                    Fu_NLG*nlg_frame.y + Fv_NLG*nlg_frame.x) + 
           me.cross(plane_lmlg_mount.pos_from(origin), 
                    Fu_LMLG*lmlg_frame.y + Fv_LMLG*lmlg_frame.x) + 
           me.cross(plane_rmlg_mount.pos_from(origin), 
                    Fu_RMLG*rmlg_frame.y + Fv_RMLG*rmlg_frame.x))

R_NLG  = (-Fu_NLG*nlg_frame.y - 
           Fv_NLG*nlg_frame.x + 
           Ft_NLG*inertial_frame.y + 
           g1)
R_LMLG = (-Fu_LMLG*lmlg_frame.y - 
           Fv_LMLG*lmlg_frame.x + 
           Ft_LMLG*inertial_frame.y + 
           g2)
R_RMLG = (-Fu_RMLG*rmlg_frame.y - 
           Fv_RMLG*rmlg_frame.x + 
           Ft_RMLG*inertial_frame.y + 
           g2)

external_forces = [(plane_CG, R_plane),
                   (plane_frame, T_plane),
                   (nlg_axis, R_NLG),
                   (plane_nlg_mount, -R_NLG),
                   (lmlg_axis, R_LMLG),
                   (plane_lmlg_mount, -R_LMLG),
                   (rmlg_axis, R_RMLG),
                   (plane_rmlg_mount, -R_RMLG)]

forces_syms = (Fu_NLG,Fv_NLG,Ft_NLG,
               Fu_LMLG,Fv_LMLG,Ft_LMLG,
               Fu_RMLG,Fv_RMLG,Ft_RMLG)

q = sp.Matrix([y,theta_x,theta_z,u1,v1,u2,v2,u3,v3])
U,repl = form_generalized_speeds_vector(t, q)

parameters = {L1:1820.0,L2:2850.0,ax:12820.0,ay:1315.0,az:0.0,bx:1280.0,by:215.0,bz:3600.0,
              g:9810.0,alpha_1:0.0,alpha_2:np.deg2rad(5.5),
              M:54.0,Jxx:1.0e9,Jxz:1.2e8,Jzz:3.8e9,m1:0.13,m2:0.57}

p_syms = list(parameters.keys())
p_vals = np.array(list(parameters.values()))

y11s =  nlg_axis.pos_from(origin).express(inertial_frame).dot(inertial_frame.y).simplify()
y12s = lmlg_axis.pos_from(origin).express(inertial_frame).dot(inertial_frame.y).simplify()
y13s = rmlg_axis.pos_from(origin).express(inertial_frame).dot(inertial_frame.y).simplify()

dy11sdt = y11s.diff(t).subs(repl)
dy12sdt = y12s.diff(t).subs(repl)
dy13sdt = y13s.diff(t).subs(repl)

y1_code = CythonMatrixGenerator([q, p_syms], [sp.Matrix([y11s,y12s,y13s])])
dy1dt_code = CythonMatrixGenerator([q, U, p_syms], [sp.Matrix([dy11sdt,dy12sdt,dy13sdt])])

calculate_y1 = y1_code.compile(tmp_dir=r"\tmp", verbose=False)
calculate_dy1dt = dy1dt_code.compile(tmp_dir=r"\tmp", verbose=False)

nlg_path = 'NLG_properties.json'
mlg_path = 'MLG_properties.json'
nlg_tyre_path = 'NLG_tyre_properties.json'
mlg_tyre_path = 'MLG_tyre_properties.json'

### ------------ Получение системы уравнений Лагранжа 2-го рода ------------ ###
LM = me.LagrangesMethod(T, 
                        q, 
                        forcelist=external_forces, 
                        frame = inertial_frame)
eqs = LM.form_lagranges_equations()

M_code = CythonMatrixGenerator([q, U, p_syms], 
                               [LM.mass_matrix.subs(repl)])
F_code = CythonMatrixGenerator([q, U, forces_syms, p_syms], 
                               [LM.forcing.subs(repl)])

eval_eom = [M_code.compile(tmp_dir=r"\tmp", verbose=False),
            F_code.compile(tmp_dir=r"\tmp", verbose=False)]
print("Numerical interpretation of the system succesfully created.")

### ------------ Численное интегрирование полученных уравнений ------------ ###
nlg = fm.DoubleChamberStrut(nlg_path)
lmlg = fm.SingleChamberStrut(mlg_path)
rmlg = fm.SingleChamberStrut(mlg_path)

nlg_tyre = fm.TyreModel(nlg_tyre_path)
lmlg_tyre = fm.TyreModel(mlg_tyre_path)
rmlg_tyre = fm.TyreModel(mlg_tyre_path)

global evaln 
evaln = 0

def plane_system(t, z, p_vals):
    global evaln
    N = int(z.shape[0]/2)
    q = z[:N]
    u = z[N:]

    y = q[0]
    tx = q[1]
    tz = q[2]
    u1 = q[3]
    v1 = q[4]
    u2 = q[5]
    v2 = q[6]
    u3 = q[7]
    v3 = q[8]

    dydt = u[0]
    dtxdt = u[1]
    dtzdt = u[2]
    du1dt = u[3]
    dv1dt = u[4]
    du2dt = u[5]
    dv2dt = u[6]
    du3dt = u[7]
    dv3dt = u[8]

    if evaln % 10000 == 0:
        print(f"eval: {evaln:10d}; t: {t:3.2f} s; q: {q}")

    y1res = np.zeros((3,),order='C',dtype=np.float64)
    dy1dtres = np.zeros((3,),order='C',dtype=np.float64)

    y1_vect = calculate_y1(q,p_vals,y1res)
    dy1dt_vect = calculate_dy1dt(q, u, p_vals, dy1dtres)

    eta1 =  nlg_tyre.calculate_deflection(y1_vect[0])
    eta2 = lmlg_tyre.calculate_deflection(y1_vect[1])
    eta3 = rmlg_tyre.calculate_deflection(y1_vect[2])

    Fu_nlg = nlg.axial_force(u1, du1dt, v1)
    Fv_nlg = nlg.reaction_bending(u1, v1, dv1dt)
    Ft_nlg = nlg_tyre.reaction_vertical(eta1, -dy1dt_vect[0])

    Fu_lmlg = lmlg.axial_force(u2, du2dt, v2)
    Fv_lmlg = lmlg.reaction_bending(u2, v2, dv2dt)
    Ft_lmlg = lmlg_tyre.reaction_vertical(eta2, -dy1dt_vect[1])

    Fu_rmlg = rmlg.axial_force(u3, du3dt, v3)
    Fv_rmlg = rmlg.reaction_bending(u3, v3, dv3dt)
    Ft_rmlg = rmlg_tyre.reaction_vertical(eta3, -dy1dt_vect[2])

    F_models = np.array([Fu_nlg,Fv_nlg,Ft_nlg,
                         Fu_lmlg,Fv_lmlg,Ft_lmlg,
                         Fu_rmlg,Fv_rmlg,Ft_rmlg]).flatten(order='C')

    Mres = np.zeros((N**2, ), order='C', dtype=np.float64)
    Fres = np.zeros((N,),order='C',dtype=np.float64)

    M = eval_eom[0](q,u,p_vals, Mres)
    F = eval_eom[1](q,u,F_models,p_vals,Fres)

    dzdt = np.empty_like(z)
    dzdt[:N] = u
    dzdt[N:] = np.linalg.solve(M, np.squeeze(F))

    evaln +=1
    return dzdt

toc = time.perf_counter()
print(f"Elapsed time is {toc-tic:3.2f} s.")
    
q0 = np.array([8000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              dtype = np.float64)
u0 = np.array([   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              dtype = np.float64)
z0 = np.hstack((q0,u0))

number_of_integration_pts = 10000
t_end = 10
tt = np.linspace(start = 0, 
                 stop = t_end, 
                 num = number_of_integration_pts, 
                 dtype = np.float64)

tic = time.perf_counter()

print("Integrating...")

result, infodict = odeint(plane_system,
                          y0 = z0,
                          t = tt,
                          args = (p_vals,),
                          rtol = 1e-5,
                          atol = 1e-5,
                          ixpr = True,
                          full_output = True,
                          tfirst = True)

toc = time.perf_counter()

print(f"Integration is done in {toc-tic:3.2f} s, plotting...")

tire_deflection_values = np.zeros((number_of_integration_pts, 3))

for i, qi in enumerate(result):
    y1res = np.zeros((3,),order='C',dtype=np.float64)
    y1_vect = calculate_y1(qi,p_vals,y1res)
    tire_deflection_values[i,0] = nlg_tyre.calculate_deflection(y1_vect[0])
    tire_deflection_values[i,1] = lmlg_tyre.calculate_deflection(y1_vect[1])
    tire_deflection_values[i,2] = rmlg_tyre.calculate_deflection(y1_vect[2])
    

### ----------------- Построение графиков заданных величин ----------------- ###

fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,dpi=300, figsize=(4,6), )
ax1.plot(tt, result.T[0])
ax1.set_xlabel(r"$t$, [s]")
ax1.set_ylabel(r"$y$, [mm]")
ax1.grid(True)

ax2.plot(tt, result.T[3])
ax2.plot(tt, result.T[5])
ax2.plot(tt, result.T[7])
ax2.legend([r"$u_{NLG}$",r"$u_{LMLG}$",r"$u_{RMLG}$"], loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
ax2.set_xlabel(r"$t$, [s]")
ax2.set_ylabel(r"$u_i$, [mm]")
ax2.grid(True)

ax3.plot(tt, tire_deflection_values[:,0])
ax3.plot(tt, tire_deflection_values[:,1])
ax3.plot(tt, tire_deflection_values[:,2])
ax3.legend([r"$\eta_{NLG}$",r"$\eta_{LMLG}$",r"$\eta_{RMLG}$"], loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
ax3.set_xlabel(r"$t$, [s]")
ax3.set_ylabel(r"$\eta_i$, [mm]")
ax3.grid(True)

ax4.plot(tt, np.rad2deg(result.T[1]))
ax4.plot(tt, np.rad2deg(result.T[2]))
ax4.legend([r"$\vartheta_x$",r"$\vartheta_z$",r"$u_{NLG}$"], loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
ax4.set_xlabel(r"$t$, [s]")
ax4.set_ylabel(r"$\vartheta_i$, [$\circ$]")
ax4.grid(True)

fig.tight_layout()

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,dpi=300, figsize=(4,4))
ax1.semilogy(tt[1:], infodict['hu'])
ax1.set_xlabel(r"$t$, [s]")
ax1.set_ylabel(r"$\eta_i$, [mm]")
ax1.grid(True)

ax2.plot(tt[1:], infodict['tcur'])
ax2.set_xlabel(r"$t$, [s]")
ax2.set_ylabel(r"$\eta_i$, [mm]")
ax2.grid(True)

ax3.plot(tt[1:], infodict['nfe'])
ax3.set_xlabel(r"$t$, [s]")
ax3.set_ylabel(r"Number of $\frac{dz}{dt} evaluation$")
ax3.grid(True)

ax4.plot(tt[1:], infodict['mused'])
ax4.set_xlabel(r"$t$, [s]")
ax4.set_ylabel(r"Method of integration")
ax4.grid(True)

fig.tight_layout()