# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 14:48:05 2023

@author: devoi
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee', 'russian-font'])
plt.rcParams.update({"font.size" : 14})

data_path = r"C:\Users\devoi\Downloads\MS-20.txt"

with open(data_path) as file:
    data = np.loadtxt(file)

data = data.T

#temp = data[0]
rho = data[1]
#visc = data[2]
beta = data[3]
temp = np.linspace(-50, 100, 10)
#beta = 1e4 * (3.95e-04 + 1.30e-6 * temp)

cm = 1/2.54

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18*cm, 7.5*cm),
                               dpi=400)

ax1.plot(temp, 1e12 * (1.040e-9 - 6.92e-13 * (temp + 273)))
ax1.plot(temp, 1e12 * (9.031e-10 - 5.673e-13 * temp))
ax1.set_xlabel(r"$T$, \textcelsius")
ax1.set_ylabel(r"$\rho$, кг\textbackslash м\textsuperscript{3}")
ax1.grid(True)
ax1.legend(["АМГ-10", "МС-20"], loc="upper right", frameon=True,
           edgecolor="white", facecolor="white", framealpha=1, fontsize=12)
ax1.margins(0.0, 0.0)

ax2.plot(temp, 1e4 * (7.50e-04 + 1.30e-6 * temp))
ax2.plot(temp, 1e4 * (5.087e-4 + 4.333e-7 * (temp + 273)))
ax2.set_xlabel(r"$T$, \textcelsius")
ax2.set_ylabel(r"$\beta$, 1\textbackslash \textcelsius $\cdot 10^4$")
ax2.grid(True)
ax2.legend(["АМГ-10", "МС-20"], frameon=True,
           edgecolor="white", facecolor="white", framealpha=1)
ax2.margins(0.0, 0.0)
ax2.ticklabel_format(axis="both", style="scientific")

fig.tight_layout()
