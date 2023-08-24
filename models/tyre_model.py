# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 23:29:41 2023

@author: devoi
"""
import json
import numpy as np
from scipy.integrate import quad


class AbstractContactPatch():
    def __init__(self, param, tyre_deflection=0.0):
        self.TOL = 1e-10
        self._param = param
        self._tyre_deflection = tyre_deflection

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, value):
        self._param = value

    @property
    def tyre_deflection(self):
        return self._tyre_deflection

    @tyre_deflection.setter
    def tyre_deflection(self, value):
        self._tyre_deflection = value if value > self.TOL else self.TOL

    @property
    def max_height(self):
        pass

    @property
    def max_width(self):
        pass

    @property
    def equation(self):
        pass

    @property
    def area(self):
        pass


class RectangleContactPatch(AbstractContactPatch):
    def __init__(self, param, tyre_deflection=0.0):
        super().__init__(param, tyre_deflection)

    @property
    def max_height(self):
        return self.param["forcing"]["R_w"]

    @property
    def max_width(self):
        R_t = self.param["forcing"]["R_t"]
        eta = self.tyre_deflection
        return np.sqrt(2 * R_t * eta - eta**2)

    @property
    def equation(self):
        def set_equation(x_coord):
            return self.max_height
        return np.vectorize(set_equation)

    @property
    def area(self):
        return 4 * self.max_height * self.max_width


class EllipseContactPatch(RectangleContactPatch):
    def __init__(self, param, tyre_deflection=0.0):
        super().__init__(param, tyre_deflection)

    @property
    def max_height(self):
        R_w = self.param["forcing"]["R_w"]
        eta = self.tyre_deflection
        return np.sqrt(2 * R_w * eta - eta**2)

    @property
    def equation(self):
        def set_equation(x_coord):
            return (self.max_height *
                    np.sqrt(1 - (x_coord / self.max_width)**2))
        return np.vectorize(set_equation)

    @property
    def area(self):
        return np.pi * self.max_height * self.max_width


class OvalContactPatch(EllipseContactPatch):
    def __init__(self, param, tyre_deflection=0.0):
        super().__init__(param, tyre_deflection)

    @property
    def equation(self):
        def set_equation(x_coord):
            R_w = self.param["forcing"]["R_w"]
            R_t = self.param["forcing"]["R_t"]
            eta = self.tyre_deflection
            return (np.sqrt(-2*R_t**2 + 2 * R_t * R_w + 2 * R_t * eta +
                            2 * R_t * np.sqrt(R_t**2 - 2 * R_t * eta +
                                              eta**2 + x_coord**2) -
                            2 * R_w * np.sqrt(R_t**2 - 2 * R_t * eta +
                                              eta**2 + x_coord**2) -
                            eta**2 - x_coord**2))
        return np.vectorize(set_equation)

    @property
    def area(self):
        area, error = quad(lambda x: 4 * self.equation(x),
                           0.0,
                           self.max_width-self.TOL)
        return area


class TyreSinglePointModel():
    patch_types = ("rectangle", "ellipse", "oval")

    def __init__(self, json_file, patch_type="ellipse"):
        with open(json_file, "r") as file:
            param = json.load(file)
        
        self.param = param
        self.param_vals = [val for val in self.param["explicit"].values()]     
        self._regularization_coeff = 1000.0

        if patch_type not in self.patch_types:
            raise ValueError(
                "Patch type must be 'rectangle', 'ellipse' or 'oval'.")
            return
        if patch_type == "rectangle":
            self.contact_patch = RectangleContactPatch(self.param)
        elif patch_type == "ellipse":
            self.contact_patch = EllipseContactPatch(self.param)
        else:
            self.contact_patch = OvalContactPatch(self.param)

    @property
    def regularization_coeff(self):
        return self._regularization_coeff

    @regularization_coeff.setter
    def regularization_coeff(self, value):
        self._regularization_coeff = value

    @property
    def heaviside_regularized(self):
        return lambda x: 0.5 * np.tanh(self.regularization_coeff * x) + 0.5

    @property
    def sign_regularized(self):
        return lambda x: np.tanh(self.regularization_coeff * x)

    def calculate_deflection(self, y1):
        eta_unconstr = self.param["forcing"]["R_t"] - y1
        return eta_unconstr * self.heaviside_regularized(eta_unconstr)
    
    def stiffness_vertical(self, tyre_deflection):
        stiffness = (tyre_deflection * 
                     self.param["forcing"]['pt'] * 
                    (2 * self.param["forcing"]['C1'] * 
                     self.param["forcing"]['pt'] + 
                     self.param["forcing"]['C2'] * tyre_deflection) / 
                    (self.param["forcing"]['C1'] * 
                     self.param["forcing"]['pt'] +
                     self.param["forcing"]['C2'] * tyre_deflection)**2)
        stiffness = max(stiffness, 0.0)
        return stiffness

    def fundamental_nat_freq(self, tyre_deflection):
        return np.sqrt(self.stiffness_vertical(tyre_deflection) /
                       self.param["explicit"]["m1"])

    def calculate_vertical_force(self, tyre_deflection, tyre_deflection_rate):
        restoring_force = (tyre_deflection**2 /
                           (self.param["forcing"]["C1"] +
                            self.param["forcing"]["C2"] *
                            tyre_deflection / self.param["forcing"]["pt"]))
        damping_force = 2 * (self.param["forcing"]["zeta"] *
                             self.param["explicit"]["m1"] *
                             self.fundamental_nat_freq(tyre_deflection) *
                             tyre_deflection_rate)
        restoring_force = max(restoring_force, 0.0)
        damping_force = max(damping_force, 0.0)
        total_force = restoring_force + damping_force
        return total_force


if __name__ == "__main__":
    json_file = "C:/Users/devoi/Thesis/Git/aircraft-taxiing-vibrations/parameters_data/MLG_tyre_properties.json"
    with open(json_file, "r") as file:
        param = json.load(file)

    rcp = RectangleContactPatch(param, 50.0)
    ecp = EllipseContactPatch(param, 50.0)
    ocp = OvalContactPatch(param, 50.0)
    etas = np.linspace(0, 200.0, 11)
    for eta in etas:
        rcp.tyre_deflection = eta
        ecp.tyre_deflection = eta
        ocp.tyre_deflection = eta
        print("--------------------------------------------------------------")
        print(f"At tyre deflecton = {eta:.2f} mm:")
        print(f"Rectangle area: {rcp.area:.2f}")
        print(f"Ellipse area: {ecp.area:.2f}")
        print(f"Oval area: {ocp.area:.2f}")
        print(f"Ellipse area rel err: {(ecp.area - ocp.area)/ocp.area:.2f}")
        print(f"Rectangle area rel err: {(rcp.area - ocp.area)/ocp.area:.2f}")
