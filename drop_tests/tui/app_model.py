import sys
import time
from enum import IntEnum

import numpy as np
from scipy.integrate import solve_ivp, _ivp

if r"C:\Users\devoi\Thesis\dev\aircraft-taxiing-vibrations\models" not in sys.path: #for .pyx import
    sys.path.append(r"C:\Users\devoi\Thesis\dev\aircraft-taxiing-vibrations\models")
if r"C:\Users\devoi\Thesis\dev\aircraft-taxiing-vibrations\utils" not in sys.path: #for .pyx import
    sys.path.append(r"C:\Users\devoi\Thesis\dev\aircraft-taxiing-vibrations\utils")

from landing_gear_model import LandingGearModel

try:
    from dll.mass_matrix_func import eval as get_mass_matrix
    from dll.forcing_vector_func import eval as get_forcing_vector
    from dll.distance_to_axle_func import eval as get_distance_to_axle
    from dll.velocity_to_axle_func import eval as get_velocity_to_axle
except ModuleNotFoundError as exc:
    sys.exit()
except ImportError as exc:
    sys.exit()

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
    ATTRIBUTE_NAMES = ("y", "u", "vertical_load")
    def __init__(self):
        super().__init__(self)
        self.t = np.array([])
        self.y = np.array([])


class DropTestModel():
    PRINTEVERY = 1000
    INIT_HEIGHT = 5.0
    KJ_COEFF = 1.0e6
    KMH2MMS_COEFF = 1000.0 / 3.6
    EVENTS = ("tyre_fall", "tyre_rise", "piston_fall", "piston_rise")
    def __init__(self, setup_dict, logger, progress_bar):
        setup_dict = self.change_strut_angle(setup_dict)

        self.landing_gear = LandingGearModel(setup_dict["strut-parameters"],
                                             setup_dict["tyre-parameters"])
        self.result = CustomResult()
        self.setup = setup_dict["test-parameters"]
        self.time_span = np.linspace(0.0, self.setup["termination-s"],
                                     self.setup["num-points"])
        self.logger = logger
        self.progress_bar = progress_bar

        self._evaln = 0

    @property
    def setup(self):
        return self._setup
    @setup.setter
    def setup(self, setup_dict):
        self._setup = setup_dict

    @property
    def initial_conditions(self):
        impact_energy = self.setup["impact-energy-kJ"] * self.KJ_COEFF
        cage_mass = self.setup["cage-mass-t"]
        init_spinup = self.setup["spinup-kmh"] * self.KMH2MMS_COEFF
        init_vel = np.sqrt(2 * impact_energy /
                           (cage_mass +
                         self.landing_gear.tyre.param["explicit"]["m1"]))
        init_length = (self.landing_gear.strut.param["explicit"]["L"] *
        np.cos(self.landing_gear.strut.param["explicit"]["alpha"]) +
        self.landing_gear.tyre.param["forcing"]["R_t"])

        init_spinup_rads = (init_spinup /
                            self.landing_gear.tyre.param["forcing"]["R_t"])

        q0 = np.array([init_length+self.INIT_HEIGHT, 0.0, 0.0, 0.0],
                      dtype = np.float64)
        u0 = np.array([-init_vel, 0.0, 0.0, init_spinup_rads, 0.0],
                      dtype = np.float64)
        z0 = np.hstack((q0, u0))
        return z0

    @staticmethod
    def change_strut_angle(setup_dict):
        new_angle = setup_dict["test-parameters"]["setup-angle"]
        setup_dict["strut-parameters"]["explicit"]["alpha"] = new_angle
        return setup_dict

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
        return [self.detect_tyre_contact(-1),
                self.detect_tyre_contact(1),
                self.detect_piston_stop(-1),
                self.detect_piston_stop(1)]

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
        Fx = 0.0
        Fy = 2 * self.landing_gear.tyre.vertical_force
        Mz = 0.0
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
            self.logger.print_message(f"eval: {self._evaln:5d}; "
                                      f"t: {t:3.2f} s; "
                                      f"q: {q}")
            self.progress_bar.update(total=100,
                                     progress=t / self.time_span[-1] * 100)
        self._evaln += 1

        M = self.compute_mass_matrix(q, U)
        F = self.compute_forcing_vector(q, U)
        self.landing_gear.tyre.slip_ratio = z[-1]

        dzdt = self.compute_dzdt(M, F, U, z)
        return dzdt

    def integrate(self):
        events_list = self.get_events_list()
        self.logger.print_message(
            '                               y        u        v       phi'
        )
        self.result = solve_ivp(self.system_eqns_rhs,
                           y0=self.initial_conditions,
                           t_span=(0, self.time_span[-1]),
                           t_eval=self.time_span,
                           events=events_list,
                           method="LSODA")

    def postprocess(self):
        N = self.landing_gear.NDOF
        num_points = self.result.y.shape[1]
        axle_position = np.zeros((num_points, ))
        horizontal_force = np.zeros((num_points, ))
        vertical_force = np.zeros((num_points, ))
        braking_torque = np.zeros((num_points, ))
        pressure = np.zeros((num_points, ))

        for i, zi in enumerate(self.result.y.T):
            qi = np.ascontiguousarray(zi[:N], dtype=np.float64)
            Ui = np.ascontiguousarray(zi[N:-1], dtype=np.float64)
            yi, dy1dti = self.compute_axle_kinematics(qi, Ui)

            self.landing_gear.tyre.calculate_deflection(yi, -dy1dti)
            self.landing_gear.strut.set_state([qi[State.U], qi[State.V],
                                               Ui[State.U], Ui[State.V]])

            axle_position[i] = yi
            horizontal_force[i] = 2 * self.landing_gear.tyre.horizontal_force
            vertical_force[i] = 2 * self.landing_gear.tyre.vertical_force
            braking_torque[i] = 0.0 #TODO: calculate in tyre model as computed attribute and return here
            pressure[i] = self.landing_gear.strut.gas_pressure

        self.result.axle_position = axle_position
        self.result.horizontal_force = horizontal_force
        self.result.vertical_force = vertical_force
        self.result.braking_torque = braking_torque
        self.result.pressure = pressure

    def process_events(self):
        self.events_dict = {
            ev_type: {"t": t_arr, "z": z_arr}
            for t_arr, z_arr, ev_type in zip(
                self.result.t_events, self.result.y_events, self.EVENTS
            )
        }

    def get_result(self):
        self.logger.print_message("Интегрирование...")
        tic = time.perf_counter()
        self.integrate()
        toc = time.perf_counter()
        self.logger.print_message(f"Интегрирование заняло {toc-tic:3.2f} s. "
                                  f"Обработка результатов...")
        self.progress_bar.update(total=100, progress=100)
        tic = time.perf_counter()
        self.postprocess()
        self.process_events()
        toc = time.perf_counter()
        self.logger.print_message("Обработка результатов завершена.")
        return self.result
