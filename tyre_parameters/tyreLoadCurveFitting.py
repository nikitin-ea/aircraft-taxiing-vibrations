import os
import natsort
import glob
import numpy as np
import scipy.optimize as opt

MAX_DEFL = 150.0

def scan_directory(path):
        os.chdir(path)
        return natsort.natsorted(glob.glob("*.txt"))
    
def load_data(path):
    filenames = scan_directory(path)
    raw_data = []
    for filename in filenames:
        with open(filename) as file:
            array = np.loadtxt(file)
        raw_data.append(array)
    return raw_data

def separate_data(raw_data):
    pressures = raw_data[-1][:,1]    
    curves = raw_data[:-1]
    return pressures, curves
    
def crop_curve(curve):
    mask = curve[:,0] <= MAX_DEFL
    return curve[mask,0], curve[mask,1]
 
def prepare_data(raw_data):
    pressures, curves = separate_data(raw_data)
    xdata = []
    ydata = []
    for curve, pressure in zip(curves, pressures):
        defl, force = crop_curve(curve)
        points = np.vstack((defl, np.full_like(defl, pressure)))
        xdata.append(points)
        ydata.append(force)
    xdata = np.hstack(xdata).T
    ydata = np.hstack(ydata).T
    return xdata, ydata, curves, pressures

def compute_R_squared(f, xdata, ydata, popt):
    residuals = ydata - f(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata - np.mean(ydata))**2)
    return 1 - (ss_res / ss_tot) 

def fit_model(model, xdata, ydata, bounds):
    try:
        popt, pcov, info, msg, ier = opt.curve_fit(model, 
                                            xdata, 
                                            ydata,
                                            #p0 = matlab_popt,
                                            bounds=bounds,
                                            method='dogbox',
                                            full_output=True,
                                            nan_policy = 'omit')
    except RuntimeError:
        print("No convergence! Parameters not found.")
        return

    print(msg)
    print(info)
    print(f"optimal values: {popt}, covariance matrix: {pcov}")
    print(f"R^2 = {compute_R_squared(model, xdata, ydata, popt):.4f}")
    return popt