# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 00:51:23 2023

@author: devoi
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.integrate import quad

import scienceplots

np.seterr(all="ignore")
plt.style.use(['science','grid','russian-font']) 

x, y = sp.symbols("x y ", real = True)
Rt, Rw, eta = sp.symbols("R_t R_w eta", positive = True)

r = Rt - Rw
a = Rw
c = Rt - eta

g = ((r**2 - a**2 + c**2 + x**2 + y**2)**2 - 4 * r**2 * (x**2 + c**2))
lc = sp.sqrt((r + a)**2 - c**2)
bc = sp.sqrt(a**2 - (c - r)**2)
#lc = sp.sqrt(2*eta)
#bc = sp.sqrt(2*Rw*eta)
g_ell = ((x / lc)**2 + (y / bc)**2 - 1)
display((g).expand().simplify())
display(g_ell.simplify())

curves = sp.Matrix(sp.solve(g, y))
curves_ell = sp.Matrix(sp.solve(g_ell, y))

params =  {Rw : 370, Rt : 610}
curves_fun = sp.lambdify([x, eta, Rw], curves, modules="numpy")
curves_ell_fun = sp.lambdify([x, eta, Rw], curves_ell, modules="numpy")


xi = np.linspace(-float(lc.subs(eta, 50).evalf(20)), 
                  float(lc.subs(eta, 50).evalf(20)), 100000)

fig, ax = plt.subplots(dpi=300)

lw = 1 

ax.plot(xi, curves_fun(xi, 0.05, params[Rw])[0].T / params[Rw],
        'k', linewidth = lw)
ax.plot(xi, curves_fun(xi, 0.05, params[Rw])[1].T / params[Rw],
        'k', linewidth = lw)
ax.plot(xi, curves_fun(xi, 0.05, params[Rw])[2].T / params[Rw],
        'k', linewidth = lw)
ax.plot(xi, curves_fun(xi, 0.05, params[Rw])[3].T / params[Rw],
        'k', linewidth = lw)

ax.plot(xi, curves_fun(xi, 0.15, params[Rw])[0].T / params[Rw],
        'k', linewidth = lw)
ax.plot(xi, curves_fun(xi, 0.15, params[Rw])[1].T / params[Rw],
        'k', linewidth = lw)
ax.plot(xi, curves_fun(xi, 0.15, params[Rw])[2].T / params[Rw],
        'k', linewidth = lw)
ax.plot(xi, curves_fun(xi, 0.15, params[Rw])[3].T / params[Rw],
        'k', linewidth = lw)

ax.plot(xi, curves_fun(xi, 0.35, params[Rw])[0].T / params[Rw],
        'k', linewidth = lw)
ax.plot(xi, curves_fun(xi, 0.35, params[Rw])[1].T / params[Rw],
        'k', linewidth = lw)
ax.plot(xi, curves_fun(xi, 0.35, params[Rw])[2].T / params[Rw], 
        'k', linewidth = lw)
ax.plot(xi, curves_fun(xi, 0.35, params[Rw])[3].T / params[Rw],
        'k', linewidth = lw)

ax.plot(xi, curves_ell_fun(xi, 0.05, params[Rw])[0].T / params[Rw], 
        '--k', linewidth = lw)
ax.plot(xi, curves_ell_fun(xi, 0.05, params[Rw])[1].T / params[Rw],
        '--k', linewidth = lw)

ax.plot(xi, curves_ell_fun(xi, 0.15, params[Rw])[0].T / params[Rw], 
        '--k', linewidth = lw)
ax.plot(xi, curves_ell_fun(xi, 0.15, params[Rw])[1].T / params[Rw],
        '--k', linewidth = lw)

ax.plot(xi, curves_ell_fun(xi, 0.35, params[Rw])[0].T / params[Rw],
        '--k', linewidth = lw)
ax.plot(xi, curves_ell_fun(xi, 0.35, params[Rw])[1].T / params[Rw],
        '--k', linewidth = lw)

ax.annotate(r"$\eta = 0.05 R_t$",
            xy=(0.305, 0.01), xycoords='data', 
            xytext=(0.32, 0.12), textcoords='data',
            arrowprops=dict(arrowstyle="-", 
                            lw=0.5,
                            color = "black",
                            connectionstyle="arc3"),
            bbox=dict(boxstyle="square,pad=0.0",
                      fc="white", 
                      ec="white", 
                      lw=0.1))

ax.annotate(r"$\eta = 0.15 R_t$",
            xy=(0.52, 0.07), xycoords='data', 
            xytext=(0.53, 0.184), textcoords='data',
            arrowprops=dict(arrowstyle="-", 
                            lw=0.5,
                            color = "black",
                            connectionstyle="arc3"),
            bbox=dict(boxstyle="square,pad=0.05",
                      fc="white", 
                      ec="white", 
                      lw=0.1))

ax.annotate(r"$\eta = 0.35 R_t$", color='white',
            xy=(0.605, 0.595), xycoords='data', 
            xytext=(0.605, 0.874), textcoords='data',
            arrowprops=dict(arrowstyle="-", 
                            lw=0.5,
                            color = "black",
                            connectionstyle="arc3"),
            bbox=dict(boxstyle="square,pad=0.0",
                      fc="white", 
                      ec="white", 
                      lw=0.1))


ax.annotate(r"$\eta = 0.35 R_t$",
            xy=(0.59, 0.75), xycoords='data', 
            xytext=(0.605, 0.874), textcoords='data',
            arrowprops=dict(arrowstyle="-", 
                            lw=0.5,
                            color = "black",
                            connectionstyle="arc3"),
            bbox=dict(boxstyle="square,pad=0.0",
                      fc="white", 
                      ec="white", 
                      lw=0.1))

ax.set_xlim([0, 0.8])
ax.set_ylim([0, 1.1])
xticks = ax.xaxis.get_major_ticks() 
xticks[0].label1.set_visible(False)

ax.set_xlabel(r"$\frac{x}{R_t}$")
ax.set_ylabel(r"$\frac{z}{R_w}$")

eta_arr = np.linspace(0.0, 0.25, 30)
Rw_arr = np.linspace(0.32, 0.38, 4)
A_persei_eta = []
A_ell_eta = []

for Rw_i in Rw_arr:
    A_persei_scratch = []
    A_ell_scratch = []
    for eta_i in eta_arr:
        lc_float = float(lc.subs(params).subs({eta : eta_i, 
                                               Rw  : Rw_i}).evalf(20))
        bc_float = float(bc.subs(params).subs({eta : eta_i, 
                                               Rw  : Rw_i}).evalf(20))
        A_persei, err_persei = quad(lambda x: 4*curves_fun(x, eta_i, Rw_i)[1], 
                                    0.0, 
                                    lc_float)
        
        A_ell = np.pi * lc_float * bc_float
        #A_ell = 2 * np.pi * eta_i * np.sqrt(Rw_i)
        
        A_persei_scratch.append(A_persei)
        A_ell_scratch.append(A_ell)
    A_persei_eta.append(A_persei_scratch)
    A_ell_eta.append(A_ell_scratch)
 
A_persei_eta = np.array(A_persei_eta)
A_ell_eta = np.array(A_ell_eta)   

fig, ax = plt.subplots(dpi = 300)

for Ap, Ae in zip(A_persei_eta, A_ell_eta):
    ax.plot(eta_arr, 100*(Ap - Ae) / Ae)
ax.set_xlabel(r"$\frac{\eta}{R_t}$")
ax.set_ylabel(r"Ошибка в площади, \%")
#ax.set_ylim([0, 5])
ax.margins(0.0)
ax.legend([f"${Rwi : 3.2f}$" for Rwi in Rw_arr],
          title=r"$\frac{{R_w}}{{R_t}}$", loc='center left', bbox_to_anchor=(1, 0.5))


bp, xp = sp.symbols("b_p x_p", positive = True)

d_eta = bp / sp.pi * sp.sqrt(2 * eta / Rt - (xp / Rt)**2)

d_eta_nd = d_eta.subs({Rt : 1}) / eta
d_eta_ndf = sp.lambdify([xp, eta, bp], d_eta_nd, modules="numpy")

def d_eta_full(xp, eta, bp):
    d_eta = d_eta_ndf(xp, eta, bp)
    if np.isnan(d_eta):
        return 0.0
    else:
        return d_eta
    
d_eta_full = np.vectorize(d_eta_full)

xi = np.linspace(-1, 1, 100000)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=300,
                               figsize=(3.5*1.2,2.625*1.2))

etas = [0.05, 0.15, 0.35]
[ax1.plot(xi, -d_eta_full(xi, etai, 0.01)) for etai in etas]
ax1.set_ylabel(r"$\frac{\Delta \eta}{\eta}$")
ax1.grid(True, which="both")
ax1.legend([rf"${etai}$" for etai in etas],
           title=r"$\frac{\eta}{R_t}$", 
           loc='center left', bbox_to_anchor=(1, 0.5))
ax1.margins(0.01)

bps = [0.01, 0.025, 0.05]
[ax2.plot(xi, -d_eta_full(xi, 0.15, bpi)) for bpi in bps]
ax2.set_xlabel(r"$\frac{x_p}{R_t}$")
ax2.set_ylabel(r"$\frac{\Delta \eta}{\eta}$")
ax2.grid(True, which="both")
ax2.legend([rf"${bpi}$" for bpi in bps],
           title=r"$\frac{b_p}{R_t}$", 
           loc='center left', bbox_to_anchor=(1, 0.5))
ax2.margins(0.01)
fig.tight_layout()