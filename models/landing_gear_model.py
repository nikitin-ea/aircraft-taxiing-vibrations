# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:03:22 2023

@author: devoi
"""
import sys
import os
import numpy as np
import sympy as sp
import sympy.physics.mechanics as me
from pydy.codegen.cython_code import CythonMatrixGenerator
from IPython.display import Math, display

if "/models" not in sys.path:
    sys.path.append("/models")
    
from models.strut_model import StrutModel
from models.tyre_model import TyreSinglePointModel

###############################################################################
me.init_vprinting(use_latex='mathjax') # Printing in IPython console

class LandingGearModel():
    num_of_dof = 4
    state_str_list = ["y", "u", "v", "phi"]
    force_str_list = ["F_s", "F_b", "F_x", "F_y", "M_z", "F_a"]

    def __init__(self, strut_param_json, tyre_param_json):
        self.strut_model = StrutModel(strut_param_json)
        self.tyre_model = TyreSinglePointModel(tyre_param_json)
        
        self.init_height = (
            self.strut_model.strut.param["explicit"]["L"] *
            np.cos(self.strut_model.strut.param["explicit"]["alpha"]) +
            self.tyre_model.param["forcing"]["R_t"])
        
        self.t = me.dynamicsymbols._t
        
        param_str_list = [param_str 
                                for param_str 
                                in self.strut_model.strut.param["explicit"].keys()]
        self.param_val_list = [param_val 
                                for param_val 
                                in self.strut_model.strut.param["explicit"].values()]
        tyre_param_str_list = [param_str 
                               for param_str 
                               in self.tyre_model.param["explicit"].keys()]
        tyre_param_val_list = [param_val 
                                for param_val 
                                in self.tyre_model.param["explicit"].values()]
        param_str_list.extend(tyre_param_str_list)
        self.param_val_list.extend(tyre_param_val_list)
        self.param_val_list = np.array(self.param_val_list)
        
        self.param_str_to_syms, self.param_sym_list = \
            self.create_str_to_sym_dict(param_str_list)
        self.state_str_to_syms, self.state_sym_list = \
            self.create_str_to_sym_dict(self.state_str_list, dynamic=True)
        self.force_str_to_syms, self.force_sym_list = \
            self.create_str_to_sym_dict(self.force_str_list)
        
        self.state_vector = sp.Matrix(self.state_sym_list)
        self.frame_inertial = me.ReferenceFrame("I")
        
        self.form_generalized_speeds_vector()
    
    @staticmethod
    def create_str_to_sym_dict(str_list, dynamic=False):
        if dynamic:
            sym_list = [me.dynamicsymbols(string) for string in str_list]
        else:
            sym_list = [sp.S(string) for string in str_list]
        return dict(zip(str_list, sym_list)), sym_list
        
    def form_generalized_speeds_vector(self):
        '''
        Form sympy.matrices.dense.MutableDenseMatrix object with generalized speeds
        components from sympy.matrices.dense.MutableDenseMatrix with generalized
        coordinates.
        
        Components denoted as U_i, i in range(1, state_vector.shape[0] + 1)

        Parameters
        ----------
        t : sympy.core.symbol.Symbol
            Time variable symbol. Must be a me.dynamicsymbols._t
        state_vector : sympy.matrices.dense.MutableDenseMatrix
            Vector of generalized coordinates as dynamic symbols.

        Returns
        -------
        u : sympy.matrices.dense.MutableDenseMatrix
            Vector of generalized speeds as dynamic symbols.
        repl : dict
            Dictionary with pairs of generalized speeds as dynamic symbols and
            corresponding time derivatives of generalized coordinates.

        '''
        self.generalized_speeds = sp.Matrix([me.dynamicsymbols(f"U_{i}") 
                                        for i in range(1, self.num_of_dof+1)])
        self.repl = {qi.diff(self.t) : ui 
                for qi, ui in zip(self.state_vector, self.generalized_speeds)}
        
    def create_additional_frames(self):
        self.frame_strut = me.ReferenceFrame("S")
        self.frame_wheel = me.ReferenceFrame("W")

        self.frame_strut.orient_axis(self.frame_inertial, 
                                     self.frame_inertial.z, 
                                     self.param_str_to_syms["alpha"])
        self.frame_wheel.orient_axis(self.frame_strut,
                                     self.frame_strut.z, 
                                     self.state_str_to_syms["phi"])
        
    def create_points(self):
        self.point_origin = me.Point("O")
        self.point_mass = me.Point("M")
        self.point_axis = me.Point("A")

        self.point_mass.set_pos(self.point_origin, 
                                self.state_str_to_syms["y"] * 
                                self.frame_inertial.y)
        self.point_axis.set_pos(self.point_mass, 
                                (-self.param_str_to_syms["L"] + 
                                  self.state_str_to_syms["u"]) * 
                                self.frame_strut.y + 
                                self.state_str_to_syms["v"] * 
                                self.frame_strut.x)

        # Initial velocity calculation
        self.frame_wheel.set_ang_vel(
            self.frame_strut, 
            self.state_str_to_syms["phi"].diff(self.t) * self.frame_wheel.z)
        self.point_origin.set_vel(self.frame_inertial, 0.0)
        self.point_mass.set_vel(
            self.frame_inertial, 
            self.point_mass.pos_from(self.point_origin).dt(self.frame_inertial))
        self.point_axis.set_vel(self.frame_inertial, 
                                self.point_axis.pos_from(self.point_origin).\
                                    dt(self.frame_inertial))
            
    def create_bodies(self):
        inertia_wheel = me.inertia(self.frame_wheel, 
                                   ixx=0, 
                                   iyy=0, 
                                   izz=self.param_str_to_syms["J1"])
    
        self.particle_mass = me.Particle("Mass",
                                    point=self.point_mass, 
                                    mass=self.param_str_to_syms["m"])
        self.rigidbody_wheel = \
            me.RigidBody("Wheels", 
                         masscenter=self.point_axis, 
                         frame=self.frame_strut, 
                         mass=self.param_str_to_syms["m1"], 
                         inertia=(inertia_wheel, self.point_axis))
        
    def compute_lagrangian(self):
        kinetic_energy_mass = \
            self.particle_mass.kinetic_energy(self.frame_inertial)
        kinetic_energy_wheel = \
            self.rigidbody_wheel.kinetic_energy(self.frame_inertial)
        kinetic_energy_wheel_spin = \
            (sp.Rational(1,2) * 
             self.param_str_to_syms["J1"] * 
             self.state_str_to_syms["phi"].diff(self.t)**2)
        self.kinetic_energy = (kinetic_energy_mass + 
                               kinetic_energy_wheel + 
                               kinetic_energy_wheel_spin)
        self.particle_mass.potential_energy = \
            (self.particle_mass.mass * 
             self.param_str_to_syms["g"] * 
             me.dot(self.point_mass.pos_from(self.point_origin),
                    self.frame_inertial.y))
        self.rigidbody_wheel.potential_energy = \
            (self.rigidbody_wheel.mass * 
             self.param_str_to_syms["g"] * 
             me.dot(self.point_axis.pos_from(self.point_origin),
                    self.frame_inertial.y))
        self.potential_energy = (self.particle_mass.potential_energy +
                                 self.rigidbody_wheel.potential_energy)
        
        self.lagrangian = self.kinetic_energy - self.potential_energy
        
    def compute_generalized_forces(self):
        force_on_point_mass = (self.force_str_to_syms["F_a"] * self.frame_inertial.y + 
                               self.force_str_to_syms["F_s"] * self.frame_strut.y + 
                               self.force_str_to_syms["F_b"] * self.frame_strut.x)
        force_on_point_axis = (self.force_str_to_syms["F_x"] * self.frame_inertial.x + 
                               self.force_str_to_syms["F_y"] * self.frame_inertial.y - 
                               self.force_str_to_syms["F_s"] * self.frame_strut.y - 
                               self.force_str_to_syms["F_b"] * self.frame_strut.x)
        moment_on_point_axis = self.force_str_to_syms["M_z"] * self.frame_wheel.z

        self.external_forces = [(self.point_mass, force_on_point_mass),
                                (self.point_axis, force_on_point_axis),
                                (self.frame_wheel, moment_on_point_axis)]
        
    def generate_lagrange_equations(self):
        self.create_additional_frames()
        self.create_points()
        self.create_bodies()
        self.compute_lagrangian()
        self.compute_generalized_forces()
        
        self.obj_lagrange_method = \
            me.LagrangesMethod(Lagrangian=self.lagrangian,
                               qs=self.state_vector,
                               forcelist=self.external_forces,
                               frame=self.frame_inertial)
        self.motion_equations = \
            self.obj_lagrange_method.form_lagranges_equations()
            
    def get_mass_matrix(self):
        return self.obj_lagrange_method.mass_matrix.subs(self.repl)
    
    def get_forcing_vector(self):
        return self.obj_lagrange_method.forcing.subs(self.repl)
    
    def get_distance_to_axle(self):
        distance_from_axis_to_origin = (
            me.dot(self.point_axis.pos_from(self.point_origin),
                   self.frame_inertial.y).simplify())

        velocity_from_axis_to_origin = \
            distance_from_axis_to_origin.diff(self.t).subs(self.repl)
        
        return distance_from_axis_to_origin, velocity_from_axis_to_origin

    def generate_c_code(self, path="/dll"):
        os.chdir(path)
        distance, velocity = self.get_distance_to_axle()
    
        y1_code = CythonMatrixGenerator([self.state_vector, 
                                         self.param_sym_list], 
                                        [sp.Matrix([distance])],
                                        prefix="distance_to_axle_func")
        dy1dt_code = CythonMatrixGenerator([self.state_vector, 
                                            self.generalized_speeds, 
                                            self.param_sym_list], 
                                           [sp.Matrix([velocity])],
                                           prefix="velocity_to_axle_func")
        M_code = CythonMatrixGenerator([self.state_vector, 
                                        self.generalized_speeds, 
                                        self.param_sym_list], 
                                       [self.get_mass_matrix()],
                                       prefix="mass_matrix_func")
        F_code = CythonMatrixGenerator([self.state_vector, 
                                        self.generalized_speeds, 
                                        self.force_sym_list, 
                                        self.param_sym_list], 
                                       [self.get_forcing_vector()],
                                       prefix="forcing_vector_func")
        
        [code_obj.write(path) for code_obj in (y1_code, 
                                               dy1dt_code, 
                                               M_code, 
                                               F_code)]

    @staticmethod
    def create_dynamic_library():
        os.system("python3 distance_to_axle_func_setup.py build_ext --inplace")
        os.system("python3 velocity_to_axle_func_setup.py build_ext --inplace")
        os.system("python3 mass_matrix_func_setup.py build_ext --inplace")
        os.system("python3 forcing_vector_func_setup.py build_ext --inplace")
        
    def lift_force(self, y):
        return (self.strut_model.strut.param["explicit"]["m"] * 
                self.strut_model.strut.param["explicit"]["g"] *
                self.strut_model.strut.heaviside_regularized(self.init_height -
                                                             y))
    
    def pprint_to_IPython_console(self):
        display(Math(r"\boxed{\mathbf{M} \mathbf{\hat{\ddot{q}}} = \mathbf{\hat{f}}}"))
        display(Math(r"\mathbf{{M}} = " + sp.latex(self.get_mass_matrix())))
        display(Math(r"\mathbf{{\hat{{f}}}} = " + sp.latex(self.get_forcing_vector())))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(['science', 'ieee', 'russian-font'])
    
    mlg_strut_path = r"C:/Users/devoi/Thesis/Git/aircraft-taxiing-vibrations/parameters_data/MLG_properties.json"
    mlg_tyre_path = r"C:/Users/devoi/Thesis/Git/aircraft-taxiing-vibrations/parameters_data/MLG_tyre_properties.json"
    nlg_strut_path = r"C:/Users/devoi/Thesis/Git/aircraft-taxiing-vibrations/parameters_data/NLG_properties.json"
    nlg_tyre_path = r"C:/Users/devoi/Thesis/Git/aircraft-taxiing-vibrations/parameters_data/NLG_tyre_properties.json"
    
    dll_path = r"C:/Users/devoi/Thesis/Git/aircraft-taxiing-vibrations/models/dll"

    nlg = LandingGearModel(nlg_strut_path, nlg_tyre_path)
    mlg = LandingGearModel(mlg_strut_path, mlg_tyre_path)
    nlg.generate_lagrange_equations()
    mlg.generate_lagrange_equations()
    
    uu1 = np.linspace(0.0, 350.0, 100)
    uu2 = np.linspace(0.0, 520.0, 100)
    dudt = np.linspace(-1000.0,1000.0,100)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(5, 4), dpi=200)
    
    ax1.plot(uu1, 1e-3 * nlg.strut_model.strut.force_gas(uu1))
    #ax1.set_title(r"Restoring force in NLG $F_{gas}(u)$")
    ax1.set_ylim((0.0, 1e-3 * nlg.strut_model.strut.force_gas(uu1[-1])))
    ax1.set_xlabel(r"$u$, мм")
    ax1.set_ylabel(r"$F_g$, кН")
    ax1.grid(True)
    ax1.margins(0.0)
    
    ax2.plot(uu2, 1e-3 * mlg.strut_model.strut.force_gas(uu2))
    #ax2.set_title(r"Restoring force in MLG $F_{gas}(u)$")
    ax2.set_ylim((0.0, 1e-3 * mlg.strut_model.strut.force_gas(uu2[-1])))
    ax2.set_xlabel(r"$u$, мм")
    ax2.set_ylabel(r"$F_g$, кН")
    ax2.grid(True)
    ax2.margins(0.0)
    
    ax3.plot(dudt, 1e-3 * nlg.strut_model.strut.force_oil(0,dudt))
    #ax3.set_title(r"Damping force in NLG $F_{oil}(\dot{u})$")
    ax3.set_ylim((1e-3 * nlg.strut_model.strut.force_oil(0,dudt[1]), 
                  1e-3 * nlg.strut_model.strut.force_oil(0,dudt[-1])))
    ax3.set_xlabel(r"$\dot{u}$, мм/с")
    ax3.set_ylabel(r"$F_o$, кН")
    ax3.grid(True)
    ax3.margins(0.0)
    
    ax4.plot(dudt, 1e-3 * mlg.strut_model.strut.force_oil(0,dudt))
    ax4.plot(dudt, 1e-3 * mlg.strut_model.strut.force_oil(500,dudt),':')
    #ax4.set_title(r"Damping force in MLG $F_{oil}(u,\dot{u})$")
    ax4.set_ylim((1e-3 * mlg.strut_model.strut.force_oil(500,dudt[1]), 
                  1e-3 * mlg.strut_model.strut.force_oil(500,dudt[-1])))
    ax4.set_xlabel(r"$\dot{u}$, мм/с")
    ax4.set_ylabel(r"$F_o$, кН")
    ax4.grid(True)
    ax4.legend([r"$u = 0$ мм", r"$u = 500$ мм"], 
               loc="lower right", frameon=True, edgecolor="white", 
               facecolor="white", framealpha=1)
    ax4.margins(0.0)
    
    fig.tight_layout()
    plt.show()