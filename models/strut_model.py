# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:59:21 2023

@author: devoi
"""
import json
import numpy as np
from scipy.optimize import fsolve

np.seterr("raise")
    
class SingleChamberStrutModel():
    regularization_coeff = 1000.0
    
    def __init__(self, param):
        self.param = param
        self.set_install_angle_in_deg(self.param["explicit"]["alpha"])
        self.piston_area = 0.25 * np.pi * self.param["forcing"]["d1"]**2
        self.force_gas_value = self.param["forcing"]["p0"] * self.piston_area
        
    def set_install_angle_in_deg(self, alpha):
        self.param["explicit"]["alpha"] = np.deg2rad(alpha)
        
    def set_regularization_coefficient(self, a):
        self.regularization_coeff = a

    def heaviside_regularized(self, x):
        return 0.5 * np.tanh(self.regularization_coeff * x) + 0.5
    
    def sign_regularized(self, x ):
        return np.tanh(self.regularization_coeff * x)
    
    def force_gas(self, u):
        preload_force = self.param["forcing"]["p0"] * self.piston_area
        new_relative_volume = np.abs(1 - self.piston_area * u / 
                                     self.param["forcing"]["V0"])
        self.force_gas_value = (preload_force * 
                                new_relative_volume**(-self.param["forcing"]["chi"]))
        return self.force_gas_value
 
    def force_oil(self, u, dudt):
        orifice_area = np.interp(u, 
                                 self.param["forcing"]["uf"], 
                                 self.param["forcing"]["f"])
        velocity_pressure = (self.param["forcing"]["rho_f"] * 
                             np.abs(dudt) * dudt)
        forward_flow_ratio  = (self.param["forcing"]["Sp"]**3 / 
                               (2 * orifice_area**2))
        backward_flow_ratio = (self.param["forcing"]["Sr"]**3 / 
                               (2 * self.param["forcing"]["fr"]**2))
        force_forward = (self.param["forcing"]["zeta1"] * 
                         velocity_pressure * 
                         forward_flow_ratio)
        force_backward = (self.param["forcing"]["zeta2"] * 
                          velocity_pressure * 
                          backward_flow_ratio)
        return (force_forward + 
                self.heaviside_regularized(-dudt)*force_backward)
 
    def force_pin_friction(self, u, dudt, v):
        reaction_1 = (self.pin_stiffness(u) * v * 
                     (self.param["forcing"]["a"] + 
                      self.param["forcing"]["b"]) / 
                     (self.param["forcing"]["b"] + u))
        reaction_2 = (self.pin_stiffness(u) * v * 
                     (self.param["forcing"]["a"] - u) / 
                     (self.param["forcing"]["b"] + u))
        total_normal_reaction = np.abs(reaction_1) + np.abs(reaction_2)
        return (self.param["forcing"]["mu_p"] * total_normal_reaction * 
                self.sign_regularized(dudt))
    
    def force_seal_friction(self, u, dudt):
        return (self.param["forcing"]["mu_s"] * 
                self.force_gas_value * 
                self.sign_regularized(dudt))
    
    def reaction_penalty(self, u):
        return (self.param["forcing"]["kappa"] * 
                u * 
                self.heaviside_regularized(-u))
    
    def reaction_bending(self, u, v, dvdt):
        return (self.pin_stiffness(u) * v + 
                self.param["forcing"]["d_bend"] * dvdt)
    
    def pin_stiffness(self, u):
        moment_of_inertia = (self.param["forcing"]["d1"]**4 - 
                             self.param["forcing"]["d2"]**4) / 64
        return (3 * np.pi * self.param["forcing"]["E"] * moment_of_inertia / 
               ((self.param["forcing"]["a"] - u)**2 * 
                (self.param["forcing"]["a"] + self.param["forcing"]["b"])))

    def axial_force(self, u, dudt, v):
        return (
            self.force_gas(u)
            + self.force_oil(u, dudt)
            + self.force_pin_friction(u, dudt, v)
            + self.force_seal_friction(u, dudt)
            + self.reaction_penalty(u)
        )
    

class DoubleChamberStrutModel(SingleChamberStrutModel):
    def __init__(self, param):
        super().__init__(param)     
        self._u_hat_init = 0.0
        self.piston_area = (0.25 * np.pi * 
                            (self.param["forcing"]["dc"]**2 - 
                            self.param["forcing"]["dp1"]**2))
        self._force_gas_vectorized = np.vectorize(self._force_gas_unvectorized, 
                                                  otypes=[float], 
                                                  excluded=["u_hat"])
    
    def force_gas(self, u):
        self.force_gas_value = self._force_gas_vectorized(u)
        return self.force_gas_value
    
    def _force_gas_unvectorized(self, u, u_hat = None):
        if u_hat is None:
            u_hat = self.calculate_u_hat(u)
        preload_force = self.param["forcing"]["p0"] * self.piston_area
        new_relative_volume = np.abs(1 - self.piston_area * 
                                     u/self.param["forcing"]["V0"] + 
                                     self.param["forcing"]["S_hat"] * 
                                     u_hat/self.param["forcing"]["V0"])
        return (preload_force * 
                new_relative_volume**(-self.param["forcing"]["chi"]))
    
    def force_gas_hat(self, u_hat):
        preload_force = (self.param["forcing"]["p0_hat"] * 
                         self.param["forcing"]["S_hat"])
        new_relative_volume = np.abs(1 - 
                                     self.param["forcing"]["S_hat"] * u_hat / 
                                     self.param["forcing"]["V0_hat"])
        return (preload_force * 
                new_relative_volume**(-self.param["forcing"]["chi_hat"]))
    
    def calculate_u_hat(self, u):
        eqn = lambda u_hat: (self._force_gas_unvectorized(u, u_hat) * 
                             self.param["forcing"]["S_hat"] / 
                             self.piston_area - 
                             self.force_gas_hat(u_hat))
        u_hat = fsolve(eqn, x0=self._u_hat_init)

        if isinstance(u_hat, np.ndarray):
            u_hat = u_hat[0]

        u_hat = max(u_hat, 0.0)
        self._u_hat_init = u_hat
        return u_hat
    
class StrutModel():
    def __init__(self, param_file_json):
        with open(param_file_json, "r") as file:
            param = json.load(file)
            
        print("loaded")
        
        if param["configuration"]["type"] == 1:
            self.strut = SingleChamberStrutModel(param)
        elif param["configuration"]["type"] == 2:
            self.strut = DoubleChamberStrutModel(param)