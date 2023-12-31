import os
import sys
import glob
import time
from enum import IntEnum

import sympy.physics.mechanics as me
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import natsort

from scipy.integrate import solve_ivp, _ivp

if r"C:\Users\devoi\Thesis\dev\aircraft-taxiing-vibrations\models" not in sys.path: #for .pyx import
    sys.path.append(r"C:\Users\devoi\Thesis\dev\aircraft-taxiing-vibrations\models")
if r"C:\Users\devoi\Thesis\dev\aircraft-taxiing-vibrations\utils" not in sys.path: #for .pyx import
    sys.path.append(r"C:\Users\devoi\Thesis\dev\aircraft-taxiing-vibrations\utils")

from text_utils import print_msg
from landing_gear_model import LandingGearModel

try:
    from dll.mass_matrix_func import eval as get_mass_matrix
    from dll.forcing_vector_func import eval as get_forcing_vector
    from dll.distance_to_axle_func import eval as get_distance_to_axle
    from dll.velocity_to_axle_func import eval as get_velocity_to_axle
except ModuleNotFoundError as exc:
    print_msg(f"{exc}: Landing gear dynamic libraries not created.")
    sys.exit()
except ImportError as exc:
    print_msg(f"{exc}: Import of landing gear dynamic libraries not possible.")
    sys.exit()

#################################global set-up#################################
me.init_vprinting(use_latex='mathjax') # Printing in IPython console
plt.style.use(['science', 'ieee', 'russian-font'])
np.set_printoptions(precision=3, suppress=True)

class State(IntEnum):
    Y = 0
    U = 1
    V = 2
    PHI = 3
    DYDT = 4
    DUDT = 5
    DVDT = 6
    DPHIDT = 7
    S = 8

class CustomResult(_ivp.ivp.OdeResult):
    TEST_DATA_NAMES = ("cg_displacement", "strut_closure", "vertical_load")
    ATTRIBUTE_NAMES = ("y", "u", "vertical_load")
    def __init__(self):
        super().__init__(self)
        self.t = np.array([])
        self.y = np.array([])
        self.t_y_test = np.array([])
        self.y_test = np.array([])
        self.t_u_test = np.array([])
        self.u_test = np.array([])
        self.t_vertical_load_test = np.array([])
        self.vertical_load_test = np.array([])
        self.t_horizontal_load_test = np.array([])
        self.horizontalload_test = np.array([])

    def load_test_data(self, filename):
        with open(filename) as file:
            temp = np.loadtxt(file)
        for data_name, attr_name in zip(self.TEST_DATA_NAMES,
                                        self.ATTRIBUTE_NAMES):
            if data_name in filename:
                if attr_name == "u":
                    self.t_u_test = temp[:,0]
                    self.u_test = temp[:,1]
                elif attr_name == "vertical_load":
                    self.t_vertical_load_test = temp[:,0]
                    self.vertical_load_test = temp[:,1]
                elif attr_name == "y":
                    self.t_y_test = temp[:,0]
                    self.y_test = temp[:,1]
                else:
                    continue

    def load_files(self, filenames):
        for filename in filenames:
            self.load_test_data(filename)


class DropTestModel():
    DEFAULT_NUMPTS = 1000
    PRINTEVERY = 1000
    INIT_HEIGHT = 5.0
    KJ_COEFF = 1e6
    EVENTS = ("tyre_fall", "tyre_rise", "piston_fall", "piston_rise")
    def __init__(self, setup_dict, t_end):
        self._num_pts = self.DEFAULT_NUMPTS
        self._setup = setup_dict
        self._evaln = 0
        self.result = CustomResult()
        self.time_span = t_end

        strut_json_name = rf"{setup_dict['lg_type']}_properties.json"
        tyre_json_name = rf"{setup_dict['lg_type']}_tyre_properties.json"
        strut_path = self.setup["path_to_params"] + "/" + strut_json_name
        tyre_path = self.setup["path_to_params"] + "/" + tyre_json_name

        self.landing_gear = LandingGearModel(strut_path, tyre_path)

    @property
    def setup(self):
        return self._setup
    @setup.setter
    def setup(self, setup_dict):
        self._setup = setup_dict

    @property
    def time_span(self):
        return self._time_span
    @time_span.setter
    def time_span(self, t_end):
        self._time_span = np.linspace(0.0, t_end, self.num_pts)

    @property
    def num_pts(self):
        return self._num_pts
    @num_pts.setter
    def num_pts(self, num):
        self._num_pts = num

    @property
    def filename(self):
        self._filename = "DT-"
        delim = "-"
        sep = "_"
        for key, value in self.setup.items():
            if key == "path_to_params":
                continue
            self._filename += key + delim + str(value) + sep
        return self._filename[:-1]

    @staticmethod
    def change_strut_angle(setup_dict):
        new_angle = setup_dict["test-parameters"]["setup-angle"]
        setup_dict["strut-parameters"]["explicit"]["alpha"] = new_angle
        return setup_dict

    @property
    def initial_conditions(self):
        impact_energy = self.setup["impact_energy_kJ"] * self.KJ_COEFF
        cage_mass = self.setup["cage_mass_t"]
        init_vel = np.sqrt(2 * impact_energy /
                           (cage_mass +
                         self.landing_gear.tyre.param["explicit"]["m1"]))
        init_length = (self.landing_gear.strut.param["explicit"]["L"] *
        np.cos(self.landing_gear.strut.param["explicit"]["alpha"]) +
        self.landing_gear.tyre.param["forcing"]["R_t"])

        q0 = np.array([init_length+self.INIT_HEIGHT, 0.0, 0.0, 0.0],
                      dtype = np.float64)
        u0 = np.array([-init_vel, 0.0, 0.0, 0.0 , 0.0],
                      dtype = np.float64)
        z0 = np.hstack((q0, u0))
        return z0

    def detect_tyre_contact(self, direction):
        def tyre_deflection(t, z):
            N = int(z.shape[0]/2)
            q = z[:N]
            y1res = np.zeros((1, ), order='C', dtype=np.float64)
            y1_vect = get_distance_to_axle(q,
                                           self.landing_gear.param_val_list,
                                           y1res)
            return (y1_vect[0] -
                    self.landing_gear.tyre.param["forcing"]["R_t"])

        tyre_event = lambda t, z: tyre_deflection(t, z)
        tyre_event.termination = False
        tyre_event.direction = direction
        return tyre_event

    def detect_piston_stop(self, direction):
        def piston_deflection(t, z):
            return z[1]
        piston_event = lambda t, z: piston_deflection(t, z)
        piston_event.termination = False
        piston_event.direction = direction
        return piston_event

    def get_events_list(self):
        tyre_event_fall = self.detect_tyre_contact(-1)
        tyre_event_rise = self.detect_tyre_contact(1)
        piston_event_fall = self.detect_piston_stop(-1)
        piston_event_rise = self.detect_piston_stop(1)
        return [tyre_event_fall,
                 tyre_event_rise,
                 piston_event_fall,
                 piston_event_rise]

    def compute_mass_matrix(self, q, U):
        Mres = np.zeros((self.landing_gear.NDOF**2, ),
                        order='C',
                        dtype=np.float64)
        mass_matrix = get_mass_matrix(q,
                                      U,
                                      self.landing_gear.param_val_list,
                                      Mres)
        return mass_matrix

    def compute_axle_kinematics(self, q, U):
        y1res = np.zeros((1, ), order='C', dtype=np.float64)
        dy1dtres = np.zeros((1, ), order='C', dtype=np.float64)
        y1 = get_distance_to_axle(q,
                                  self.landing_gear.param_val_list,
                                  y1res)[0]
        dy1dt = get_velocity_to_axle(q,
                                     U,
                                     self.landing_gear.param_val_list,
                                     dy1dtres)[0]
        return y1, dy1dt

    def compute_forcing_vector(self, q, U):
        alpha = self.landing_gear.strut.param["explicit"]["alpha"]
        y1, dy1dt = self.compute_axle_kinematics(q, U)
        self.landing_gear.tyre.calculate_deflection(y1, dy1dt)
        self.landing_gear.tyre.ang_vel = U[State.PHI]
        self.landing_gear.tyre.horz_vel = (U[State.U] * np.sin(alpha) +
                                           U[State.V] * np.cos(alpha))

        self.landing_gear.strut.set_state([q[State.U],
                                           q[State.V],
                                           U[State.U],
                                           U[State.V]])

        Fsum = self.landing_gear.strut.force_total_axial
        Fv = self.landing_gear.strut.force_bending
        Fx = (self.landing_gear.WHEELS_NUM *
              self.landing_gear.tyre.horizontal_force)
        Fy = (self.landing_gear.WHEELS_NUM *
              self.landing_gear.tyre.vertical_force)
        Mz = (self.landing_gear.WHEELS_NUM *
              self.landing_gear.tyre.braking_torque)
        Fa = self.landing_gear.lift_force(q[State.Y])

        F_models = np.array([Fsum, Fv, Fx, Fy, Mz, Fa]).flatten(order='C')
        Fres = np.zeros((self.landing_gear.NDOF, ),
                        order='C',
                        dtype=np.float64)
        forcing_vector = get_forcing_vector(q,
                                            U,
                                            F_models,
                                            self.landing_gear.param_val_list,
                                            Fres)
        return forcing_vector

    def compute_dzdt(self, M, F, U, z):
        N = self.landing_gear.NDOF

        dzdt = np.empty_like(z)
        dzdt[:N] = U
        dzdt[N:-1] = np.linalg.solve(M, np.squeeze(F))

        alpha = self.landing_gear.strut.param["explicit"]["alpha"]
        d2vhdt2 = (dzdt[State.DUDT] * np.sin(alpha) +
                   dzdt[State.DVDT] * np.cos(alpha))
        dzdt[-1] = self.landing_gear.tyre.calculate_slip_ratio_diff(d2vhdt2)
        return dzdt

    def system_eqns_rhs(self, t, z):
        '''
        Calculation of time derivatives of state vector components.

        Parameters
        ----------
        t : Float
            Time.
        z : numpy.ndarray
            Numpy array of state vector components values.
        landing_gear : LandingGearModel
            Object that contains all force models and parameters of landing gear.

        Returns
        -------
        dzdt : numpy.ndarray
            Time derivatives of state vector components.

        '''
        N = self.landing_gear.NDOF
        q = z[:N]
        U = z[N:-1]

        if self._evaln % self.PRINTEVERY == 0:
            print_msg(f"eval: {self._evaln:5d}; "
                      f"t: {t:3.2f} s; "
                      f"q: {q}")

        self._evaln += 1

        M = self.compute_mass_matrix(q, U)
        F = self.compute_forcing_vector(q, U)
        self.landing_gear.tyre.slip_ratio = z[-1]

        dzdt = self.compute_dzdt(M, F, U, z)
        return dzdt

    def integrate(self):
        events_list = self.get_events_list()
        print_msg("                               y        u        v        φ")
        self.result = solve_ivp(self.system_eqns_rhs,
                           y0=self.initial_conditions,
                           t_span=(0, self.time_span[-1]),
                           t_eval=self.time_span,
                           events=events_list,
                           method="LSODA")

    def postprocess(self):
        self.result.axle_position = np.zeros((self.result.y.shape[1], ))
        self.result.tyre_deflection = np.zeros((self.result.y.shape[1], ))
        self.result.vertical_force = np.zeros((self.result.y.shape[1], ))
        for i, zi in enumerate(self.result.y.T):
            N = int(zi.shape[0]/2)
            qi = np.ascontiguousarray(zi[:N], dtype=np.float64)
            Ui = np.ascontiguousarray(zi[N:-1], dtype=np.float64)
            y1res = np.zeros((3,), order='C', dtype=np.float64)
            dy1dtres = np.zeros((3,), order='C', dtype=np.float64)
            self.result.axle_position[i] = get_distance_to_axle(qi,
                self.landing_gear.param_val_list,
                y1res)[0]
            dy1dt = get_velocity_to_axle(qi,
                                         Ui,
                                         self.landing_gear.param_val_list,
                                         dy1dtres)[0]
            self.landing_gear.tyre.calculate_deflection(
            self.result.axle_position[i], -dy1dt)
            self.result.tyre_deflection[i] = self.landing_gear.tyre.deflection

            self.result.vertical_force[i] = (2 *
                self.landing_gear.tyre.vertical_force)

    def process_events(self):
        self.events_dict = {
            ev_type: {"t": t_arr, "z": z_arr}
            for t_arr, z_arr, ev_type in zip(
                self.result.t_events, self.result.y_events, self.EVENTS
            )
        }

    def get_result(self):
        print_msg("Integrating...")
        tic = time.perf_counter()
        self.integrate()
        toc = time.perf_counter()
        print_msg(f"Integration took {toc-tic:3.2f} s. Postproccessing...")
        tic = time.perf_counter()
        self.postprocess()
        self.process_events()
        toc = time.perf_counter()
        print_msg("Postproccessing is done.")
        print_msg(f"Postprocessing took {toc-tic:3.2f} s.")
        return self.result

    def load_file(self, filename):
        try:
            temp = np.load(filename)
        except ValueError as exc:
            print(f"{exc}: cannot load binary file!")
            return
        print(f"Загружена реализация размерностью "
                f"{temp[1:].shape[0]}×{temp[1:].shape[1]} "
                f"({(temp[1:].size * temp[1:].itemsize / 1024):3.2f} кБ).")
        self.result.t = temp[0]
        self.result.y = temp[1:]

    def save_data(self, path):
        data = np.vstack((self.result.t, self.result.y))

        with open(path + "\\" + self.filename + ".npy", 'wb') as file:
            np.save(file, data)

###############################################################################
if __name__ == "__main__":
    setup = {"path_to_params": r"C:/Users/devoi/Thesis/dev/aircraft-taxiing-vibrations/parameters_data",
             "lg_type": "MLG",
             "impact_energy_kJ": 242,
             "cage_mass_t": 33.9,
             "angle_deg": 2.5}
    drop_test = DropTestModel(setup, 1.5)
