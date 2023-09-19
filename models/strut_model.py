# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:59:21 2023

@author: devoi
"""
import json
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import pchip_interpolate
from enum import IntEnum, Enum

np.seterr("raise")

class StrutTypes(IntEnum):
    SINGLE_CHAMBER = 1
    DOUBLE_CHAMBER = 2


class AMG10(Enum):
    NAME = "AMG-10"
    RHO_COEFFS = [1.04e-9, -6.92e-13]
    BETA_COEFFS = [3.95e-04, 1.30e-6]


class MS20(Enum):
    NAME = "MS-20"
    RHO_COEFFS = [1.04e-9, -6.92e-13]
    BETA_COEFFS = [3.95e-04, 1.30e-6]


class OilModel:
    INIT_TEMP = 293.0

    def __init__(self, oil_enum):
        self.temperature = self.INIT_TEMP
        self.name = oil_enum.NAME.value
        self.rho_coeffs = oil_enum.RHO_COEFFS.value
        self.beta_coeffs = oil_enum.BETA_COEFFS.value

    @property
    def temperature(self):
        return self._temperature
    @temperature.setter
    def temperature(self, value):
        if value > 0.0:
            self._temperature = value
        else:
            raise ValueError("Temperature must be positive!")

    @property
    def density(self):
        density = 0.0
        for power, coeff in enumerate(self.rho_coeffs):
            density += coeff * self.temperature**power
        return density

    @property
    def expansion_coeff(self):
        expansion_coeff = 0.0
        for power, coeff in enumerate(self.beta_coeffs):
            expansion_coeff += coeff * self.temperature**power
        return expansion_coeff


class SingleChamberStrutModel:
    REG_COEFF = 1000.0
    INIT_TEMP = 293.0
    def __init__(self, param, oil):
        self.type = StrutTypes.SINGLE_CHAMBER
        self.param = param

        self.oil = OilModel(oil)
        self.set_geometry()
        self.set_initial_state()

        self.oil.temperature = self.temperature

    def set_geometry(self):
        self.set_install_angle_in_deg(self.param["explicit"]["alpha"])
        self.piston_area = 0.25 * np.pi * self.param["forcing"]["d1"]**2
        self.moment_of_inertia = (self.param["forcing"]["d1"]**4 -
                                  self.param["forcing"]["d2"]**4) / 64

    def set_initial_state(self):
        self._temperature = self.INIT_TEMP
        self.set_state([0.0, 0.0, 0.0, 0.0])

    @property
    def deflection(self):
        return self._deflection
    @deflection.setter
    def deflection(self, value):
        self._deflection = value
        self._invalidate_due_to_deflection()

    @property
    def deflection_vel(self):
        return self._deflection_vel
    @deflection_vel.setter
    def deflection_vel(self, value):
        self._deflection_vel = value
        self._invalidate_due_to_deflection_vel()

    @property
    def flexure(self):
        return self._flexure
    @flexure.setter
    def flexure(self, value):
        self._flexure = value
        self._invalidate_due_to_flexure()

    @property
    def flexure_vel(self):
        return self._flexure_vel
    @flexure_vel.setter
    def flexure_vel(self, value):
        self._flexure_vel = value
        self._invalidate_due_to_flexure_vel()

    @property
    def temperature(self):
        return self._temperature
    @temperature.setter
    def temperature(self, value):
        if value > 0.0:
            self._temperature = value
            self.oil.temperature = value
            self._invalidate_due_to_temperature()
        else:
            raise ValueError("Temperature must be positive!")

    @property
    def init_gas_volume(self):
        if self._init_gas_volume is None:
            self._init_gas_volume = self.calculate_init_gas_volume()
        return self._init_gas_volume

    def calculate_init_gas_volume(self):
        volume = (self.param["forcing"]["V0"] -
                  self.oil.expansion_coeff *
                  (self.temperature - self.INIT_TEMP) *
                  self.param["forcing"]["V0l"])
        return volume

    @property
    def gas_volume(self):
        if self._gas_volume is None:
            self._gas_volume = self.calculate_gas_volume()
        return self._gas_volume

    def calculate_gas_volume(self):
        gas_volume = self.init_gas_volume - self.piston_area * self.deflection
        return gas_volume

    @property
    def init_gas_pressure(self):
        if self._init_gas_pressure is None:
            self._init_gas_pressure = self.calculate_init_gas_pressure()
        return self._init_gas_pressure

    def calculate_init_gas_pressure(self):
        temperature_ratio = self.temperature / self.INIT_TEMP
        volume_ratio = self.param["forcing"]["V0"] / self.init_gas_volume
        pressure = (self.param["forcing"]["p0"] *
                    temperature_ratio *
                    volume_ratio)
        return pressure

    @property
    def gas_pressure(self):
        if self._gas_pressure is None:
            self._gas_pressure = self.calculate_gas_pressure()
        return self._gas_pressure

    def calculate_gas_pressure(self):
        pressure = (self.init_gas_pressure *
                    np.abs(self.init_gas_volume /
                    self.gas_volume)**self.param["forcing"]["chi"])
        return pressure

    @property
    def preload_force(self):
        if self._preload_force is None:
            self._preload_force = self.calculate_preload_force()
        return self._preload_force

    def calculate_preload_force(self):
        force = self.gas_pressure * self.piston_area
        return force

    @property
    def fixed_bearing_reaction(self):
        if self._fixed_bearing_reaction is None:
            self._fixed_bearing_reaction = self.calculate_fixed_bearing_reaction()
        return self._fixed_bearing_reaction

    def calculate_fixed_bearing_reaction(self):
        reaction_1 = (self.pin_stiffness * self.flexure *
                     (self.param["forcing"]["a"] +
                      self.param["forcing"]["b"]) /
                     (self.param["forcing"]["b"] + self.deflection))
        return reaction_1

    @property
    def sliding_bearing_reaction(self):
        if self._sliding_bearing_reaction is None:
            self._sliding_bearing_reaction = self.calculate_sliding_bearing_reaction()
        return self._sliding_bearing_reaction

    def calculate_sliding_bearing_reaction(self):
        reaction_2 = (self.pin_stiffness * self.flexure *
                     (self.param["forcing"]["a"] - self.deflection) /
                     (self.param["forcing"]["b"] + self.deflection))
        return reaction_2

    @property
    def pin_stiffness(self):
        if self._pin_stiffness is None:
            self._pin_stiffness = self.calculate_pin_stiffness()
        return self._pin_stiffness

    def calculate_pin_stiffness(self):
        stiffness = (3 * np.pi * self.param["forcing"]["E"] *
                     self.moment_of_inertia /
                     ((self.param["forcing"]["a"] - self.deflection)**2 *
                      (self.param["forcing"]["a"] +
                       self.param["forcing"]["b"])))
        return stiffness

    @property
    def force_gas(self):
        if self._force_gas is None:
            self._force_gas = self.calculate_force_gas()
        return self._force_gas

    def calculate_force_gas(self):
        force = self.gas_pressure * self.piston_area
        return force

    @property
    def force_oil(self):
        if self._force_oil is None:
            self._force_oil = self.calculate_force_oil()
        return self._force_oil

    def calculate_force_oil(self):
        zeta_1 = self.param["forcing"]["zeta1"]
        zeta_2 = self.param["forcing"]["zeta2"]
        oil_density = self.oil.density
        orifice_area = np.interp(self.deflection,
                                 self.param["forcing"]["uf"],
                                 self.param["forcing"]["f"])
        velocity_pressure = (oil_density *
                             np.abs(self.deflection_vel) *
                             self.deflection_vel)
        forward_flow_ratio  = (self.param["forcing"]["Sp"]**3 /
                               (2 * orifice_area**2))
        backward_flow_ratio = (self.param["forcing"]["Sr"]**3 /
                               (2 * self.param["forcing"]["fr"]**2))

        force_forward = zeta_1 * forward_flow_ratio * velocity_pressure
        force_backward = ((zeta_2 * backward_flow_ratio * velocity_pressure) *
                          self.heaviside(-self.deflection_vel))
        force = force_forward + force_backward
        return force

    @property
    def force_bearing_friction(self):
        if self._force_bearing_friction is None:
            self._force_bearing_friction = self.calculate_force_bearing_friction()
        return self._force_bearing_friction

    def calculate_force_bearing_friction(self):
        total_reaction = (np.abs(self.fixed_bearing_reaction) +
                          np.abs(self.sliding_bearing_reaction))
        force = (self.param["forcing"]["mu_p"] * total_reaction *
                 self.sign(self.deflection_vel))
        return force

    @property
    def force_seal_friction(self):
        if self._force_seal_friction is None:
            self._force_seal_friction = self.calculate_force_seal_friction()
        return self._force_seal_friction

    def calculate_force_seal_friction(self):
        force = (self.param["forcing"]["mu_s"] * self.force_gas *
                self.sign(self.deflection_vel))
        return force

    @property
    def force_penalty(self):
        if self._force_penalty is None:
            self._force_penalty = self.calculate_force_penalty()
        return self._force_penalty

    def calculate_force_penalty(self):
        force = (self.param["forcing"]["kappa"] *
                self.deflection *
                self.heaviside(-self.deflection))
        return force

    @property
    def force_bending(self):
        if self._force_bending is None:
            self._force_bending = self.calculate_force_bending()
        return self._force_bending

    def calculate_force_bending(self):
        force = (self.pin_stiffness * self.flexure +
                self.param["forcing"]["d_bend"] * self.flexure_vel)
        return force

    @property
    def force_total_axial(self):
        force = (self.force_gas +
                 self.force_oil +
                 self.force_bearing_friction +
                 self.force_seal_friction +
                 self.force_penalty)
        return force

    def _invalidate_due_to_temperature(self):
        self._init_gas_pressure = None
        self._init_gas_volume = None
        self._gas_pressure = None
        self._gas_volume = None
        self._preload_force = None
        self._pin_stiffness = None
        self._force_gas = None
        self._force_oil = None
        self._force_bearing_friction = None
        self._force_seal_friction = None
        self._force_penalty = None
        self._force_bending = None
        self._fixed_bearing_reaction = None
        self._sliding_bearing_reaction = None

    def _invalidate_due_to_deflection(self):
        self._init_gas_pressure = None
        self._init_gas_volume = None
        self._gas_pressure = None
        self._gas_volume = None
        self._pin_stiffness = None
        self._force_gas = None
        self._force_oil = None
        self._force_bearing_friction = None
        self._force_seal_friction = None
        self._force_penalty = None
        self._force_bending = None
        self._fixed_bearing_reaction = None
        self._sliding_bearing_reaction = None

    def _invalidate_due_to_deflection_vel(self):
        self._force_oil = None
        self._force_bearing_friction = None
        self._force_seal_friction = None
        self._force_bending = None

    def _invalidate_due_to_flexure(self):
        self._pin_stiffness = None
        self._force_bearing_friction = None
        self._force_seal_friction = None
        self._force_penalty = None
        self._force_bending = None
        self._fixed_bearing_reaction = None
        self._sliding_bearing_reaction = None

    def _invalidate_due_to_flexure_vel(self):
        self._force_bending = None

    def set_state(self, state):
        self.deflection = state[0]
        self.flexure = state[1]
        self.deflection_vel = state[2]
        self.flexure_vel = state[3]

    def set_install_angle_in_deg(self, alpha):
        self.param["explicit"]["alpha"] = np.deg2rad(alpha)

    def heaviside(self, x):
        return 0.5 * np.tanh(self.REG_COEFF * x) + 0.5

    def sign(self, x ):
        return np.tanh(self.REG_COEFF * x)


class DoubleChamberStrutModel(SingleChamberStrutModel):
    def __init__(self, param, oil):
        super().__init__(param, oil)
        self._fl_piston_deflection = None
        self.precomputing()

    def set_geometry(self):
        self.set_install_angle_in_deg(self.param["explicit"]["alpha"])
        self.piston_area = (0.25 * np.pi *
                            (self.param["forcing"]["dc"]**2 -
                            self.param["forcing"]["dp1"]**2))
        self.moment_of_inertia = (self.param["forcing"]["d1"]**4 -
                                  self.param["forcing"]["d2"]**4) / 64

    def precomputing(self):
        u_max = self.init_gas_volume / self.piston_area
        self._u_hat_init = 0.0
        self.DEFL_PTS = np.linspace(0.0, u_max, 1000)
        self.FLP_DEFL_PTS = np.vectorize(self.calculate_u_hat, 
                                         otypes=[float])(self.DEFL_PTS)
        self.CH1_PRESS = self.calculate_1st_chamber_pressure(self.DEFL_PTS,
                                                             self.FLP_DEFL_PTS)
        self.CH2_PRESS = self.calculate_2nd_chamber_pressure(self.FLP_DEFL_PTS)
        print("Double chamber oleo-pneumatic strut initialised.")

    def calculate_u_hat(self, deflection=0.0):
        eqn = lambda u_hat: (self.calculate_2nd_chamber_pressure(u_hat) -
                             self.calculate_1st_chamber_pressure(deflection,
                                                                 u_hat))
        u_hat = fsolve(eqn, x0=self._u_hat_init)

        if isinstance(u_hat, np.ndarray):
            u_hat = u_hat[0]

        u_hat = max(u_hat, 0.0)
        self._u_hat_init = u_hat
        return u_hat

    def calculate_1st_chamber_pressure(self, deflection, u_hat):
        volume = (self.init_gas_volume - self.piston_area * deflection +
                 self.param["forcing"]["S_hat"] * u_hat)
        pressure = (self.init_gas_pressure *
                    np.abs(self.init_gas_volume / volume)**
                    self.param["forcing"]["chi"])
        return pressure

    def calculate_2nd_chamber_pressure(self, u_hat):
        volume = (self.param["forcing"]["V0_hat"] -
                  self.param["forcing"]["S_hat"] * u_hat)
        temp_ratio = self.temperature / self.INIT_TEMP
        pressure = (self.param["forcing"]["p0_hat"] * temp_ratio *
                    np.abs(self.param["forcing"]["V0_hat"] / volume)**
                    self.param["forcing"]["chi_hat"])
        return pressure

    @property
    def fl_piston_deflection(self):
        if self._fl_piston_deflection is None:
            self._fl_piston_deflection = pchip_interpolate(self.DEFL_PTS,
                                                        self.FLP_DEFL_PTS,
                                                        self.deflection)
        return self._fl_piston_deflection

    def calculate_gas_volume(self):
        volume = (self.init_gas_volume - self.piston_area * self.deflection +
                 self.param["forcing"]["S_hat"] * self.fl_piston_deflection)
        return volume

    def calculate_gas_pressure(self):
        pressure = self.calculate_1st_chamber_pressure(self.deflection,
                                                       self.fl_piston_deflection)
        return pressure

    def _invalidate_due_to_temperature(self):
        super()._invalidate_due_to_temperature()
        self._fl_piston_deflection = None
        self.precomputing()

    def _invalidate_due_to_deflection(self):
        super()._invalidate_due_to_deflection()
        self._fl_piston_deflection = None

class StrutModel:
    def __new__(cls, param_file_json, *args):
        if isinstance(param_file_json, str):
            with open(param_file_json, "r") as file:
                param = json.load(file)
        else:
            param = param_file_json

        strut_type = param["configuration"]["type"]

        if not strut_type:
            raise KeyError("Unknown strut type!")

        if strut_type == StrutTypes.SINGLE_CHAMBER:
            return SingleChamberStrutModel(param, *args)
        elif strut_type == StrutTypes.DOUBLE_CHAMBER:
            return DoubleChamberStrutModel(param, *args)
        else:
            raise ValueError(param["configuration"]["type"])


def get_strut_model(param_file_json, *args):
    with open(param_file_json, "r") as file:
        param = json.load(file)

    strut_type = param["configuration"]["type"]

    if not strut_type:
        raise KeyError("Unknown strut type!")

    if strut_type == StrutTypes.SINGLE_CHAMBER:
        return SingleChamberStrutModel(param, *args)
    elif strut_type == StrutTypes.DOUBLE_CHAMBER:
        return DoubleChamberStrutModel(param, *args)
    else:
        raise ValueError(param["configuration"]["type"])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scienceplots
    cm = 1/2.54
    plt.style.use(['science', 'ieee', 'russian-font'])
    plt.rcParams.update({"font.size": 14})

    nlg_strut_path = r"C:/Users/devoi/Thesis/dev/aircraft-taxiing-vibrations/parameters_data/NLG_properties.json"
    mlg_strut_path = r"C:/Users/devoi/Thesis/dev/aircraft-taxiing-vibrations/parameters_data/MLG_properties.json"

    nlg_strut_model = get_strut_model(nlg_strut_path, AMG10)
    mlg_strut_model = get_strut_model(mlg_strut_path, AMG10)

    uu1 = np.linspace(0.0, 340.0, 1000)
    uu2 = np.linspace(0.0, 520.0, 1000)
    dudt = np.linspace(-1200.0,1200.0,1000)

    nlg_strut_model.deflection = uu1
    mlg_strut_model.deflection = uu2
    nlg_strut_model.deflection_vel = dudt
    mlg_strut_model.deflection_vel = dudt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2,
                                                 figsize=(18.5*cm, 2*7.5*cm),
                                                 dpi=400)

    ax1.plot(uu1, 1e-3 * nlg_strut_model.force_gas)
    ax1.set_title(r"\textit{а}")
    ax1.set_ylim((0.0, 1e-3 * np.max(nlg_strut_model.force_gas)))
    ax1.set_xlabel(r"$u$, мм")
    ax1.set_ylabel(r"$F_g$, кН")
    ax1.grid(True)
    ax1.margins(0.0)

    ax2.plot(uu2, 1e-3 * mlg_strut_model.force_gas)
    ax2.set_title(r"\textit{б}")
    ax2.set_ylim((0.0, 1e-3 * np.max(mlg_strut_model.force_gas)))
    ax2.set_xlabel(r"$u$, мм")
    ax2.set_ylabel(r"$F_g$, кН")
    ax2.grid(True)
    ax2.margins(0.0)

    ax3.plot(dudt, 1e-3 * nlg_strut_model.force_oil)
    ax3.set_title(r"\textit{в}")
    ax3.set_ylim((1e-3 * np.min(nlg_strut_model.force_oil),
                  1e-3 * np.max(nlg_strut_model.force_oil)))
    ax3.set_xlabel(r"$\dot{u}$, мм/с")
    ax3.set_ylabel(r"$F_o$, кН")
    ax3.grid(True)
    ax3.margins(0.0)

    mlg_strut_model.deflection = 0.0 * np.ones_like(dudt)
    ax4.plot(dudt, 1e-3 * mlg_strut_model.force_oil)
    mlg_strut_model.deflection = 500.0 * np.ones_like(dudt)
    ax4.plot(dudt, 1e-3 * mlg_strut_model.force_oil,':')
    ax4.set_title(r"\textit{г}")
    ax4.set_ylim((1e-3 * np.min(mlg_strut_model.force_oil),
                  1e-3 * np.max(mlg_strut_model.force_oil)))
    ax4.set_xlabel(r"$\dot{u}$, мм/с")
    ax4.set_ylabel(r"$F_o$, кН")
    ax4.grid(True)
    ax4.legend([r"$u = 0$ мм", r"$u = 500$ мм"],
               loc="lower right", frameon=True, edgecolor="white",
               facecolor="white", framealpha=1)
    ax4.margins(0.0)

    fig.tight_layout()
    plt.show()

    temps = [223.0, 273.0, 293.0, 343.0]
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(18*cm,7.5*cm),
                                   dpi=400)

    nlg_strut_model.deflection = uu1
    mlg_strut_model.deflection = uu2
    nlg_strut_model.deflection_vel = np.linspace(-1200.0,1200.0,1000)
    mlg_strut_model.deflection_vel = np.linspace(-1200.0,1200.0,1000)

    for temp in temps:
        nlg_strut_model.temperature = temp
        mlg_strut_model.temperature = temp
        ax1.plot(nlg_strut_model.deflection,
                 1e-3 * nlg_strut_model.force_gas)
        ax2.plot(mlg_strut_model.deflection,
                 1e-3 * mlg_strut_model.force_gas)

    ax1.set_ylim((0 *1e-3 * np.min(nlg_strut_model.force_gas),
                  1e-3 * np.max(nlg_strut_model.force_gas)))
    ax1.set_xlabel(r"$u$, мм")
    ax1.set_ylabel(r"$F_g$, кН")
    ax1.grid(True)
    ax1.legend([rf"${(temp - 273):+05.1f}$ \textcelsius" for temp in temps],
              loc="upper left", frameon=True, edgecolor="white",
              facecolor="white", framealpha=1)
    ax1.margins(0.0)

    ax2.set_ylim((0 * 1e-3 * np.min(mlg_strut_model.force_gas),
                  1e-3 * np.max(mlg_strut_model.force_gas)))
    ax2.set_xlabel(r"$u$, мм")
    ax2.set_ylabel(r"$F_g$, кН")
    ax2.grid(True)
    ax2.legend([rf"${(temp - 273):+05.1f}$ \textcelsius" for temp in temps],
              loc="upper left", frameon=True, edgecolor="white",
              facecolor="white", framealpha=1)
    ax2.margins(0.0)

    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize = (9*cm, 7.5*cm))
    ax.plot(mlg_strut_model.param["forcing"]["uf"], mlg_strut_model.param["forcing"]["f"])
    ax.set_xlabel(r"$u$, мм")
    ax.set_ylabel(r"$f_h$, мм\textsuperscript{2}")
    ax.grid(True)
    ax.set_ylim([0.0, 1.1 * max(mlg_strut_model.param["forcing"]["f"])])
    ax.margins(0.0, 0.2)

    fig.tight_layout()
