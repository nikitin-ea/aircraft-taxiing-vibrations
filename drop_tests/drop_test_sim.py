from datetime import datetime

def print_msg(text):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(dt_string + " " + text)

import os    
import sys 
import glob   
import time
import sympy.physics.mechanics as me
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import natsort

from scipy.integrate import solve_ivp, _ivp

if "/models/dll" not in sys.path: #for .pyx import
    sys.path.append("/models/dll")

from models.landing_gear_model import LandingGearModel
 
try:
    from models.dll.mass_matrix_func import eval as get_mass_matrix 
    from models.dll.forcing_vector_func import eval as get_forcing_vector 
    from models.dll.distance_to_axle_func import eval as get_distance_to_axle
    from models.dll.velocity_to_axle_func import eval as get_velocity_to_axle
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
        
    def load_test_data(self, filename):
        with open(filename) as file:
            temp = np.loadtxt(file)
        for data_name, attr_name in zip(self.TEST_DATA_NAMES, 
                                        self.ATTRIBUTE_NAMES):
            if data_name in filename:   
                if attr_name == "y":
                    self.t_y_test = temp[:,0]
                    self.y_test = temp[:,1]
                if attr_name == "u":
                    self.t_u_test = temp[:,0]
                    self.u_test = temp[:,1]
                if attr_name == "vertical_load":
                    self.t_vertical_load_test = temp[:,0]
                    self.vertical_load_test = temp[:,1]
                else:
                    continue
                
    def load_files(self, filenames):
        for filename in filenames:
            self.load_test_data(filename)
        

class DropTestModelCase():
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
         
    @property
    def initial_conditions(self):
        impact_energy = self.setup["impact_energy_kJ"] * self.KJ_COEFF
        cage_mass = self.setup["cage_mass_t"]
        init_vel = np.sqrt(2 * impact_energy / 
                           (cage_mass + 
                         self.landing_gear.tyre_model.param["explicit"]["m1"]))
        init_length = (self.landing_gear.strut_model.strut.param["explicit"]["L"] * 
        np.cos(self.landing_gear.strut_model.strut.param["explicit"]["alpha"]) +
        self.landing_gear.tyre_model.param["forcing"]["R_t"])

        q0 = np.array([init_length+self.INIT_HEIGHT, 0.0, 0.0, 0.0],
                      dtype = np.float64)
        u0 = np.array([-init_vel, 0.0, 0.0, 0.0],
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
                    self.landing_gear.tyre_model.param["forcing"]["R_t"])
    
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
        # Initialization
        N = int(z.shape[0]/2)
        q = z[:N]
        U = z[N:]

        y = q[0]
        u = q[1]
        v = q[2]
        #phi = q[3]
        #dydt = u[0]
        dudt = U[1]
        dvdt = U[2]
        #dphidt = u[3]

        y1res = np.zeros((1, ), order='C', dtype=np.float64)
        dy1dtres = np.zeros((1, ), order='C', dtype=np.float64)
        Mres = np.zeros((N**2, ), order='C', dtype=np.float64)
        Fres = np.zeros((N, ), order='C', dtype=np.float64)

        # Using C programs for calculation of wheels position
        y1_vect = get_distance_to_axle(q, 
                                       self.landing_gear.param_val_list, 
                                       y1res)
        dy1dt_vect = get_velocity_to_axle(q, 
                                          U, 
                                          self.landing_gear.param_val_list, 
                                          dy1dtres)
        M = get_mass_matrix(q, 
                            U, 
                            self.landing_gear.param_val_list, 
                            Mres)

        # Evaluation of force models
        eta = self.landing_gear.tyre_model.calculate_deflection(y1_vect[0])

        Fsum = self.landing_gear.strut_model.strut.axial_force(u, dudt, v)
        Fv = self.landing_gear.strut_model.strut.reaction_bending(u, v, dvdt)
        Fx = 0.0
        Fy = 2 * self.landing_gear.tyre_model.calculate_vertical_force(eta, 
                                                                -dy1dt_vect[0])
        Mz = 0.0
        Fa = self.landing_gear.lift_force(y)

        F_models = np.array([Fsum, Fv, Fx, Fy, Mz, Fa]).flatten(order='C')

        # Using C programs for calculation of mass matrix and forcing vectors
        F = get_forcing_vector(q, 
                               U, 
                               F_models, 
                               self.landing_gear.param_val_list, 
                               Fres)

        # Evaluation of time derivatives vector
        dzdt = np.empty_like(z)
        dzdt[:N] = U
        dzdt[N:] = np.linalg.solve(M, np.squeeze(F))
        
        # Iterations counting 
        if self._evaln % self.PRINTEVERY == 0:
            print_msg(f"eval: {self._evaln:5d}; t: {t:3.2f} s; q: {q}")
        self._evaln += 1
        return dzdt
    
    def integrate(self):
        events_list = self.get_events_list()
        print_msg("                               y        u        v       phi")
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
            Ui = np.ascontiguousarray(zi[N:], dtype=np.float64)
            y1res = np.zeros((3,), order='C', dtype=np.float64)
            dy1dtres = np.zeros((3,), order='C', dtype=np.float64)
            self.result.axle_position[i] = get_distance_to_axle(qi, 
                                                    self.landing_gear.param_val_list, 
                                                    y1res)[0]
            dy1dt = get_velocity_to_axle(qi, 
                                         Ui, 
                                         self.landing_gear.param_val_list, 
                                         dy1dtres)[0]
            self.result.tyre_deflection[i] = self.landing_gear.tyre_model.calculate_deflection(
                self.result.axle_position[i])
            self.result.vertical_force[i] = (2 * 
                self.landing_gear.tyre_model.calculate_vertical_force(
                self.result.tyre_deflection[i], -dy1dt))
    
    def process_events(self):
        self.events_dict = {}
        
        for t_arr, z_arr, ev_type in zip(self.result.t_events, 
                                         self.result.y_events, 
                                         self.EVENTS):
            self.events_dict[ev_type] = {"t": t_arr, "z": z_arr}
            
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
    
    @staticmethod
    def set_xmargin(ax, left=0.0, right=0.3):
        ax.set_xmargin(0)
        ax.autoscale_view()
        lim = ax.get_xlim()
        delta = np.diff(lim)
        left = lim[0] - delta*left
        right = lim[1] + delta*right
        ax.set_xlim(left,right)
        
    @staticmethod
    def draw_subplot(ax, xdata, ydata, style=None, text=None, limits=None):
        if style is None:
            ax.plot(xdata, ydata)
        else:
            ax.plot(xdata, ydata, style)
        ax.grid(True)
        
        if text is not None:
            ax.set_title(f"{text['title']}")
            ax.set_xlabel(f"{text['xlabel']}")
            ax.set_ylabel(f"{text['ylabel']}")
              
        if limits is not None:
            try:
                ax.set_xlim(limits["xlim"])
            except KeyError:
                pass
            try:
                ax.set_ylim(limits["ylim"])
            except KeyError:
                pass
    
    def plot_results(self):
        tic = time.perf_counter()
        print_msg("Creating plots...")
        
        t_plot_end = self.events_dict["tyre_fall"]["t"][1]
        
        self.fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, 
                                                                 ncols=2,
                                                                 figsize=(4,5),
                                                                 dpi=400)
        y_text = {"title": r"\textit{а}",
                  "xlabel": r"$t$, c",
                  "ylabel": r"$y$, мм"}
        u_text = {"title": r"\textit{б}",
                  "xlabel": r"$t$, c",
                  "ylabel": r"$u$, мм"}
        v_text = {"title": r"\textit{в}",
                  "xlabel": r"$t$, c",
                  "ylabel": r"$v$, мм"}
        Fyt_text = {"title": r"\textit{г}",
                    "xlabel": r"$t$, c",
                    "ylabel": r"$F_y$, кН"}
        Fyu_text = {"title": r"\textit{д}",
                    "xlabel": r"$u$, мм",
                    "ylabel": r"$F_y$, кН"}
        Fyy_text = {"title": r"\textit{е}",
                    "xlabel": r"$y_0-y$, мм",
                    "ylabel": r"$F_y$, кН"}
        
        Fyy_lims = {"xlim": [0.0, np.max(self.result.y[0][0] - self.result.y[0])]}
        
        [self.set_xmargin(ax, left=0.0, right=0.05) for ax in (ax1, ax2, ax3, 
                                                               ax4, ax5, ax6)]
        
        self.draw_subplot(ax1, self.result.t, self.result.y[0], text=y_text)
        self.draw_subplot(ax2, self.result.t, self.result.y[1], text=u_text)
        self.draw_subplot(ax3, self.result.t, self.result.y[2], text=v_text)
        self.draw_subplot(ax4, self.result.t, 1e-3 * self.result.vertical_force, 
                          text=Fyt_text)
        self.draw_subplot(ax5, self.result.y[1][self.result.t<t_plot_end], 
                          1e-3 * self.result.vertical_force[self.result.t<t_plot_end], 
                          text=Fyu_text)
        self.draw_subplot(ax6, self.result.y[0][0] - 
                          self.result.y[0][self.result.t<t_plot_end], 
                          1e-3 * self.result.vertical_force[self.result.t<t_plot_end], 
                          text=Fyy_text,
                          limits=Fyy_lims)
        
        u_max = np.max(self.result.y[1][self.result.t<t_plot_end])
        uu = np.linspace(0.0, u_max, 100)
        self.draw_subplot(ax5, uu,
                     1e-3 * self.landing_gear.strut_model.strut.force_gas(uu),
                     style="--")
           
        self.fig.tight_layout()
        toc = time.perf_counter()
        print_msg(f"Plots creating took {toc-tic:3.2f} s. Rendering...")
        tic = time.perf_counter()    
        plt.show()
        toc = time.perf_counter()
        print_msg(f"Image rendering took {toc-tic:3.2f} s. Saving...")    
    
    def save_figure(self, path):
        self.fig.savefig(path + "\\" + self.filename + ".svg", 
                         transparent=True)
        
    def save_data(self, path):
        data = np.vstack((self.result.t, self.result.y))

        with open(path + "\\" + self.filename + ".npy", 'wb') as file:
            np.save(file, data)
            
    @staticmethod
    def scan_directory(path):
        os.chdir(path)
        filenames = natsort.natsorted(glob.glob("*.npy"))
        return filenames
    
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
    
###############################################################################
if __name__ == "__main__":
    setup = {"path_to_params": r"C:/Users/devoi/Thesis/Git/aircraft-taxiing-vibrations/parameters_data",
             "lg_type": "MLG",
             "impact_energy_kJ": 242,
             "cage_mass_t": 33.9,
             "angle_deg": 2.5}
    dtms = DropTestModelCase(setup, 1.5)