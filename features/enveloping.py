# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:53:53 2023

@author: devoi
"""
import sympy.physics.mechanics as me
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from IPython.display import display
from scipy.integrate import odeint
from scipy.signal import find_peaks
import time
import json
import os
from pydy.codegen.cython_code import CythonMatrixGenerator

import scienceplots

from force_models import TyreModel

SIZE = 10
plt.style.use(['science', 'grid', 'russian-font'])
plt.rc('font', size = SIZE)
plt.rc('axes', titlesize = SIZE)
plt.rc('axes', labelsize = SIZE)
plt.rc('xtick', labelsize = SIZE)
plt.rc('ytick', labelsize = SIZE)
me.init_vprinting(use_latex='mathjax')


savepath = r"C:\Users\devoi\Thesis\Git\time-series-manager\time-series-manager\simple_enveloping1"

def form_generalized_speeds_vector(t,q):
    n = q.shape[0]
    u = sp.Matrix([me.dynamicsymbols(f"U_{i}") for i in range(1,n+1)])
    repl = {qi.diff(t) : ui for qi, ui in zip(q, u)}
    return u, repl

def generate_system(param_dict):
    tyre = TyreModel(tyre_model_path)

    m, Jz, L, g, Fy, dF = sp.symbols(r'm J_z L g F_y \Delta_F', positive = True)
    y, phi = me.dynamicsymbols('y varphi')
    t = me.dynamicsymbols._t

    inertial_frame = me.ReferenceFrame("N")
    wheel_frame = me.ReferenceFrame("W")
    wheel_frame.orient(inertial_frame, 'Axis', (phi, inertial_frame.z))
    
    origin = me.Point("O")
    axle_location = origin.locatenew("M", (L + y) * inertial_frame.y)
    contact_point = axle_location.locatenew("C", (-L + y) * inertial_frame.y)
    
    origin.set_vel(inertial_frame, 0)
    axle_location.set_vel(inertial_frame, 
                          axle_location.pos_from(origin).dt(inertial_frame))
    
    wheel_inertia = Jz*me.outer(wheel_frame.z, wheel_frame.z)
    
    wheel = me.RigidBody("Wheel", 
                         axle_location, 
                         wheel_frame, 
                         m, 
                         (wheel_inertia, axle_location))
    
    display(wheel.kinetic_energy(inertial_frame))
    
    wheel_potential_energy = m * g * me.dot(axle_location.pos_from(origin),
                                             inertial_frame.y)
    wheel.potential_energy = wheel_potential_energy
    display(wheel.potential_energy)
    R = Fy * inertial_frame.y + dF * inertial_frame.y
    display(R)
    
    lagrangian = wheel.kinetic_energy(inertial_frame) - wheel.potential_energy
    q = sp.Matrix([y, phi])
    U,repl = form_generalized_speeds_vector(t, q)
    
    forces = [(contact_point, R)]
    
    LM = me.LagrangesMethod(lagrangian, 
                            q, 
                            forcelist = forces, 
                            frame = inertial_frame)
    eqns = LM.form_lagranges_equations()

    parameters = {m : param_dict['mass'], g : 9810.0, 
                  Jz: 1.0e5, L : param_dict['R_t']}
    p_syms = list(parameters.keys())
    p_vals = np.array(list(parameters.values()))
    eta = (L - y)*(0.5 * sp.tanh(1000 * (L - y)) + 0.5)
    F_syms = [Fy, dF]
    
    M_code = CythonMatrixGenerator([q, p_syms], 
                                   [LM.mass_matrix.subs(repl)])
    F_code = CythonMatrixGenerator([F_syms, p_syms], 
                                   [LM.forcing.subs(repl)])

    eval_eom = [M_code.compile(tmp_dir=r"\tmp", verbose=False),
                F_code.compile(tmp_dir=r"\tmp", verbose=False)]
    
    return eval_eom, tyre, p_vals

def model_eqns(t, z, p_vals, exc_dict, tyre, eval_eom):
    global x0, H, V
    
    N = int(z.shape[0]/2)
    q = z[:N]
    u = z[N:]
    eta = tyre.calculate_deflection(q[0])
    
    
    deta = tyre.pothole_excitation(t, 
                                   exc_dict["V"], 
                                   eta, 
                                   exc_dict["H"], 
                                   exc_dict["b"])
    
    Fy = tyre.reaction_vertical(eta + deta, -u[0])
    dF = 0.0
    
    F_vals = np.array([Fy, dF], order = 'C')
    
    Mres = np.zeros((N*N,),order='C',dtype=np.float64)
    Fres = np.zeros((N,),order='C',dtype=np.float64)
    
    M = eval_eom[0](q,p_vals,Mres)
    F = eval_eom[1](F_vals,p_vals,Fres)
    
    dzdt = np.zeros_like(z)
    dzdt[:N] = u
    dzdt[N:] = np.linalg.solve(M, np.squeeze(F))
    
    return dzdt

def integrate_system(z0, p_vals, exc_dict, tyre, eval_eom):
    dt = exc_dict["b"] / (6 * exc_dict["V"])
    t_steady = 5.0
    print(f"Calculated time step dt = {dt:3.5f} sec.")
    t_end = 20 * exc_dict["H"] / exc_dict["V"] + t_steady 
    tt = np.arange(start = 0, 
                   stop = t_end, 
                   step = dt, 
                   dtype = np.float64)
    
    tic = time.perf_counter()
    
    print(f'Integrating for V = {0.001 * exc_dict["V"]:3.2f} m/s...')
    
    result, infodict = odeint(model_eqns,
                              y0 = z0,
                              t = tt,
                              args = (p_vals, exc_dict, tyre, eval_eom),
                              rtol = 1e-7,
                              atol = 1e-6,
                              ixpr = True,
                              full_output = True,
                              tfirst = True)
    
    toc = time.perf_counter()
    
    print('\a')
    os.chdir(savepath)
    header = "y phi dydt dphidt"
    np.savetxt(f"{(exc_dict['V'] / 1000 * 3.6):3.2f}.txt", 
               np.hstack((np.expand_dims(tt, axis=0).T, 
                          result)),
               header=header)
    
    print(f"Integration is done in {toc-tic:3.2f} s.")
    return tt, result

def plot(tt, result, tyre):
    eta = tyre.calculate_deflection(result.T[0])
    deta = []
    for ti, etai in zip(tt, eta):
      deta.append(tyre.pothole_excitation(ti, V, etai, H, 50))  
      
    deta = np.array(deta)
    
    t1 = 0.7*tt[-1]
    t2 = tt[-1]
    
    mask = np.logical_and(tt > t1, tt < t2)
    cm = 1 / 2.54
    rat = 2 / 3
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols = 2, 
                                                 nrows = 2, 
                                                 figsize = (2* 8.5 * cm, 
                                                            2 * rat * 8.5 * cm), 
                                                 dpi=300)
    
    ax1.plot(tt[mask], result.T[0][mask],'k')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.set_title(r"\textit{а}")
    ax1.set_xlabel(r"$t$, [с]")
    ax1.set_ylabel(r"$y$, [мм]")
    ax1.margins(0, 0.05)
    
    ax2.plot(tt[mask], eta[mask],'k')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_title(r"\textit{б}")
    ax2.set_xlabel(r"$t$, [с]")
    ax2.set_ylabel(r"$\eta$, [мм]")
    ax2.margins(0, 0.05)
    
    ax3.plot(tt[mask], 
             1e-3 * tyre.reaction_vertical(eta[mask], 
                                           -result.T[2][mask]),'k')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.set_xlabel(r"$t$, [с]")
    ax3.set_ylabel(r"$F_y$, [кН]")
    ax3.set_title(r"\textit{в}")
    ax3.margins(0, 0.05)
    
    ax4.plot(tt[mask], deta[mask],'k')
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax4.set_xlabel(r"$t$, [с]")
    ax4.set_ylabel(r"$\Delta \eta$, [мм]")
    ax4.set_title(r"\textit{г}")
    ax4.margins(0, 0.05)
    
    fig.tight_layout()
    
def poincare_analysis(param_dict, exc_dict, v_start, v_stop, points):
    eval_eom, tyre, p_vals = generate_system(param_dict)
    z0 = np.array([309.1, 0.0 , 0.0, 0.0])
    
    peaks_list = []
    vv = np.linspace(v_start, v_stop, points)
    
    for vi in vv:
        exc_dict['V'] = vi
        tt, result = integrate_system(z0, p_vals, exc_dict, tyre, eval_eom)
        peaks, props = find_peaks(result.T[0])
        peaks_list.append(result.T[0][peaks])
    return vv, peaks_list, tt, result

#################
tyre_model_path = 'NLG_tyre_properties_modified.json'
with open(tyre_model_path, 'r') as file:
    param_dict = json.load(file)
    
exc_dict = {"V" : 1e3, "H" : 6000.0, "b" : 50.0}

joined_dict = param_dict | exc_dict | {"analysis" : "V"}

with open(savepath + r"parameters.json", 'w') as file:
    json.dump(joined_dict, file, indent=4)

v_start = 54397.95918368
v_stop = 70e3
points = 11

vv, peaks_list, tt, result = poincare_analysis(param_dict,
                                   exc_dict,
                                   v_start, 
                                   v_stop, 
                                   points)

fig, ax = plt.subplots(dpi=300)
for vi, peaks in zip(vv, peaks_list):
    ax.plot(vi * np.ones(peaks.shape[0]), peaks, '.k', markersize=2)