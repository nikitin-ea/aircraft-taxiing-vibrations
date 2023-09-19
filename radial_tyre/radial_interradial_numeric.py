# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:31:56 2023

@author: devoi
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.optimize import fsolve

import scienceplots

SIZE = 10

plt.style.use(['science', 'grid', 'russian-font'])
plt.rc('font', size = SIZE)
plt.rc('axes', titlesize = SIZE)
plt.rc('axes', labelsize = SIZE)
plt.rc('xtick', labelsize = SIZE)
plt.rc('ytick', labelsize = SIZE)
plt.rc('legend', fontsize = 0.8 * SIZE)
np.set_printoptions(precision=5)

class BadalamentiTyre():
    def __init__(self, tyre_props):
        self.tyre_props = tyre_props
        self.deflections = np.zeros(self.tyre_props["springs_num"])
        self.axle_position = self.tyre_props["tyre_radius"]
        
    def get_force_radial(self):
        self.force_radial_array = np.zeros_like(self.deflections)
        for i in range(self.tyre_props["springs_num"]):
            radial_force = (self.tyre_props["radial_stiffness"] * self.deflections[i] +
                            self.tyre_props["radial_quad_stiffness"] * self.deflections[i]**2)
            if i == 0:  
                interradial_force = (self.tyre_props["interradial_stiffness"] * 
                                     (self.deflections[i] - self.deflections[i+1]))
            elif i > 0 and i < tyre_props["springs_num"] - 1:
                interradial_force = (self.tyre_props["interradial_stiffness"] * 
                                     (2*self.deflections[i] - 
                                      self.deflections[i-1] -
                                      self.deflections[i+1]))
            else:
                interradial_force = (self.tyre_props["interradial_stiffness"] * 
                                     (self.deflections[i] - self.deflections[i-1]))
            self.force_radial_array[i] = radial_force + interradial_force
            
    def residual(self, vertical_load):
        pass
        
        
        

def get_initial_contact_angle(axle_position, tyre_props):
    central_deflection = tyre_props["tyre_radius"] - axle_position
    return np.arcsin((tyre_props["tyre_radius"] - central_deflection) /
                     tyre_props["tyre_radius"])

def get_deflections(axle_position, tyre_props):
    
    if axle_position > tyre_props["tyre_radius"]:
        return np.zeros(tyre_props["springs_num"])
    
    thetas = np.linspace(0.0, 2 * np.pi, tyre_props["springs_num"])
    deflections = np.zeros(tyre_props["springs_num"])
    
    for i, theta in enumerate(thetas):
        if np.abs(np.sin(theta)) < 1e-5:
            deflections[i] = 0.0
            continue
        deflections[i] = (tyre_props["tyre_radius"]  - axle_position / 
                          np.sin(theta))
        if deflections[i] < 0.0:
            deflections[i] = 0.0
            #print(f"deflection at angle {theta:3.2f}: {deflections[i]:3.2f}")
    return deflections

def displace_springs(deflections, tyre_props):
    init_positions = (tyre_props["tyre_radius"] * 
                      np.ones(tyre_props["springs_num"]))
    new_positions = init_positions - deflections
    return new_positions

def draw(axle_position, deflections, tyre_props, axes, axle_radius_ratio = 0.5):
    new_positions = displace_springs(deflections, tyre_props)
    thetas = np.linspace(0.0, 2 * np.pi, tyre_props["springs_num"])
    
    axle_radius = axle_radius_ratio * tyre_props["tyre_radius"]
    
    axes.plot(new_positions * np.cos(thetas), -new_positions * np.sin(thetas),\
              lw=0.2)
    axes.axhline(-axle_position, lw=0.2)
    
    for i,theta in enumerate(thetas):
        axes.plot([axle_radius* np.cos(theta), 
                   new_positions[i] * np.cos(theta)],
                  [-axle_radius * np.sin(theta),
                   -new_positions[i] * np.sin(theta)], 
                  'k',
                  lw = 0.5)
    
    circle = patches.Circle([0.0, 0.0], radius=axle_radius,
                            fc="white",
                            ec="black")
    axes.add_patch(circle)
    
    axes.set_aspect("equal")
    
def residual(axle_position, vertical_load, tyre_props):
    thetas = np.linspace(0.0, 2 * np.pi, tyre_props["springs_num"])
    deflections = get_deflections(axle_position, tyre_props)
    radial_forces = get_force_radial(deflections, tyre_props) 
    vertical_reaction = np.sum(radial_forces * np.sin(thetas))
    return vertical_load - vertical_reaction

def find_equilibrium(vertical_load, init_pos = None, tyre_props = {}):
    epsilon = 1e-3
    
    if init_pos is None:
        init_pos = tyre_props["tyre_radius"]
    
    result = fsolve(residual, 
                    init_pos, 
                    args=(vertical_load, tyre_props), 
                    xtol=epsilon,
                    full_output=True)
    print(f"axle_position = {result[0]}, infodict = {result[1]}")
    return result[0], result[1]

def load_curve(max_load, points, tyre_props):
    vertical_loads = np.linspace(10.0, max_load, points, tyre_props)
    axle_positions = np.zeros_like(vertical_loads)
    
    init_axle_position = tyre_props["tyre_radius"] - 10.0
    for i, vertical_load in enumerate(vertical_loads):
        axle_position, infodict = find_equilibrium(vertical_load,
                                                   init_pos = init_axle_position,
                                                   tyre_props = tyre_props)
        init_axle_position = axle_position
        axle_positions[i] = axle_position
        
    return tyre_props["tyre_radius"] - axle_positions, vertical_loads



#tyre_props = {"springs_num" : 35,
              #"tyre_radius" : 707.136,
              #"radial_stiffness" : 145.5,
              #"interradial_stiffness" : 1886.0,
              #"radial_quad_stiffness" : 0
              #}

tyre_props = {"springs_num" : 17 + 1,
              "tyre_radius" : 707.136,
              "radial_stiffness" : 211.0,
              "interradial_stiffness" : 0,
              "radial_quad_stiffness" : 0.0
              }


with open("Goodyear C-130.txt") as file:
    data = np.loadtxt(file, dtype=np.float64)

max_load = data.T[0][-1]
points = 20
axle_positions, vertical_loads =  load_curve(max_load, points, tyre_props)

fig, ax = plt.subplots(dpi = 300)
ax.plot(1e-3 * data.T[0], data.T[1], '--k')
ax.plot(1e-3 * vertical_loads, axle_positions, 'k-o',
        markersize=3)
ax.set_title("""Нагрузочная характеристика шины 
                 Goodyear 20-20 Type III
                 $p_t = 8.6$ атм, $R_t$ = 707 мм""")
ax.set_xlabel("Вертикальная реакция, кН")
ax.set_ylabel("Обжатие шины, мм")
ax.legend(["Эксперимент", "Квадратичная-линейная"], 
          loc="best", frameon=True, edgecolor="white", 
          facecolor="white", framealpha=1, ncols=1)

fig, ax = plt.subplots(dpi = 300)   
draw(tyre_props["tyre_radius"] - 250.0, tyre_props, ax)    