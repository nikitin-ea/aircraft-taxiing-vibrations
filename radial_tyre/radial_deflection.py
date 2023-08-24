# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 21:50:06 2023

@author: devoi
"""

import sympy as sp
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import scienceplots

sp.init_printing(use_latex=True)
plt.style.use(['science', 'grid', 'russian-font']) 

Rt, eta, c1, theta, xb, b,theta1,theta2 = sp.symbols("R_t eta c_1 theta x_b b theta_1 theta_2", positive = True)

theta_c = sp.asin((Rt - eta) / Rt)
theta_c_series = theta_c.series(x = eta, n = 1).removeO()

Fv_integrand = c1 * (Rt * (1 - sp.sin(theta)) + eta)
Fv = 1 / sp.pi *sp.integrate(Fv_integrand, 
                             (theta, theta1, theta2))
Fv = Fv.simplify()
display(Fv)
repl = {Rt : 675, c1 : 4700, b : 10}

Fv_fun = sp.lambdify((eta, theta1, theta2), 
                     Fv.subs(repl), 
                     modules="numpy")

theta_c_fun = sp.lambdify((eta), 
                          theta_c.subs(repl), 
                          modules="numpy")
    
def pothole_load_deflection(eta, xb = 0.0, width = 10.0, Rt = 675.0):
    theta_c = theta_c_fun(eta)
    theta_1 = np.arctan2((Rt - eta), (xb + width / 2))
    theta_2 = np.arctan2((Rt - eta), (xb - width / 2))

    theta_1 = min(theta_1, np.pi / 2)
    theta_2 = max(theta_2, theta_c)
    theta_1 = min(theta_1, theta_2)
    Fv_pothole = Fv_fun(eta, theta_1, theta_2)
    Fv_full = 2 * Fv_fun(eta, theta_c, np.pi / 2)

    Fv_sum = Fv_full - Fv_pothole
    return max(Fv_sum, 0.0)

Fv_pothole = np.vectorize(pothole_load_deflection,
                          otypes = ['float'], 
                          excluded = ['xb', 'Rt'])

Fv_full = lambda eta: Fv_fun(eta, theta_c_fun(eta), np.pi - theta_c_fun(eta))

ee = np.linspace(0.0, 100.0, 1000)

fig, ax = plt.subplots(dpi = 500, figsize = (3, 3))
ax.plot(ee, 1e-3 * Fv_full(ee))
ax.plot(ee, 1e-3 * Fv_pothole(ee, xb = 0.0, width = 30.0, Rt = 675.0))
ax.plot(ee, 1e-3 * Fv_pothole(ee, xb = 60.0, width = 30.0, Rt = 675.0))
ax.plot(ee, 1e-3 * Fv_pothole(ee, xb = 200.0, width = 30.0, Rt = 675.0))
ax.set_xlabel(r"$\eta$, [мм]")
ax.set_ylabel(r"$F_y$, [кН]")
ax.grid(True)
ax.legend([r"Без шва", 
           r"$x_b = 0.0$ мм", 
           r"$x_b = 60.0$ мм", 
           r"$x_b = 200.0$ мм"], 
           frameon=True, edgecolor="white", 
           facecolor="white", framealpha=1)
plt.show()

        