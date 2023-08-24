# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 20:49:04 2023

@author: devoi
"""
import numpy as np
from scipy.signal import sawtooth 

class Pothole():
    def __init__(self, width, spacing, time=0.0, speed=0.0):
        self.width = width
        self.spacing = spacing
        self._time = time
        self._speed = speed
    
    @property
    def time(self):
        return self._time
    
    @time.setter
    def time(self, value):
        self._time = value 
    
    @property
    def speed(self):
        return self._speed
    
    @speed.setter
    def speed(self, value):
        self._speed = value 
      
    @property
    def position(self):
        return (0.5 * self.spacing * 
                sawtooth(2 * np.pi * self.speed * self.time / self.spacing))
    
    def excitation(self, contact_patch):
        if np.abs(self.position) < contact_patch.max_width:
            return 0.0
        eta = contact_patch.tyre_deflection
        R_t = contact_patch.tyre_parameters["R_t"]
        return (R_t - eta - 
                0.5 * np.sqrt(4 * (R_t**2 - eta)**2 - 
                              4 * np.sqrt(eta) * self.width * 
                              np.sqrt(2 * R_t - eta) - 
                              self.width**2)) 