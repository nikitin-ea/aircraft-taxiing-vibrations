import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intrp
#import scienceplots

from cycler import cycler
from tyreLoadCurveFitting import separate_data, crop_curve

#plt.style.use(['science', 'russian-font', 'no-latex'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
color_cycle = cycler(c=colors)

GRID_PTS = 100

def create_grid(xdata):
    range_x = np.linspace(0.0, np.max(xdata[:,0]), GRID_PTS)
    range_y = np.linspace(np.min(xdata[:,1]), np.max(xdata[:,1]), GRID_PTS)
    return np.meshgrid(range_x, range_y)

def interpolate_data(xdata, ydata):
    grid_x, grid_y = create_grid(xdata)
    return grid_x, grid_y, intrp.griddata(xdata, ydata, (grid_x, grid_y), method='linear')

def plot3d_scatter_and_wireframe(ax, xdata, ydata):
    grid_x, grid_y, grid_z = interpolate_data(xdata, ydata)
    ax.plot_wireframe(grid_x, grid_y, grid_z, cstride=5, rstride=5, alpha=0.7)
    ax.scatter(*xdata.T, ydata, 'o', facecolor='w', edgecolor='b')
    plt.draw()
    
def plot2d_load_curves(ax, raw_data, model, popt):   
    pressures, curves = separate_data(raw_data)
    for curve, pressure, style in zip(curves, pressures, color_cycle):
        defl, force = crop_curve(curve)
        points = np.vstack((defl, np.full_like(defl, pressure)))
        ax.plot(defl, force, label = rf"$p_t={pressure:.2f}$ МПа", linestyle = ':', **style)
        ax.plot(defl, model(points.T, *popt), **style)
    ax.legend()
    ax.margins(0.0)
    plt.draw()
    
def plot_all(xdata, ydata, raw_data, model, popt):
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111)
    plot3d_scatter_and_wireframe(ax1, xdata, ydata)
    plot2d_load_curves(ax2, raw_data, model, popt)   
    plt.show()