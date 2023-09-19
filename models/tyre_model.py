# -*- coding: utf-8 -*-
"""Tyre with single point contact model, used for carrying out virtual drop
tests of landing gears, dynamic aircraft taxiing or vehicle simulation.

@author: Egor Nikitin
@copyright: ©2023 PJSC Yakovlev
@credits: Egor Niktin
@license: MIT
@version 1.0
@maintainer: Egor Nikitin
@email: devoitene@gmail.com
@status: Production
"""

import json
import numpy as np
from scipy.integrate import quad

class AbstractContactPatch():
    """Base class for contact patch object.

    Attributes
    ----------
    TOL : float
        Tolerance of contact patch parameters calculation.

    Properties
    ----------
    param : dict
        Dictionary of tyre parameters.
    tyre_deflection : float, optional
        Deflection of a tyre. The default is 0.0.
    max_width : float
        Maximum width of contact patch (at its center.)
    max_length : float
        Maximum length of contact patch (at its center.)
    equation : callable[x_coord : float] -> float
        Equation for computing the width of contact patch depending
        of coordinate ``x_coord`` along its length.
    area : float
        Area of contact patch.

    Methods
    -------
    None.
    """
    TOL = 1e-8 #!!! Class attribute
    def __init__(self, param: dict, tyre_deflection: float = 0.0):
        """Constructor of base class for contact patch object.

        Parameters
        ----------
        param : dict
            Dictionaty of tyre parameters.
        tyre_deflection : float, optional
            Deflection of a tyre. The default is 0.0.

        Returns
        -------
        None.

        """
        self._param = param
        self._tyre_deflection = tyre_deflection

    @property
    def param(self) -> dict:
        """Dictionary of tyre parameters.

        :getter: Returns dictionary of tyre parameters.
        :setter: Sets dictionary of tyre parameters.
        :type: dict
        """
        return self._param
    @param.setter
    def param(self, value: dict):
        if isinstance(value, dict):
            self._param = value
        else:
            raise TypeError

    @property
    def tyre_deflection(self) -> float:
        """Deflection of tyre.

        :getter: Returns value of deflection of tyre.
        :setter: Sets value of deflection of tyre.
        :type: float
        """
        return self._tyre_deflection
    @tyre_deflection.setter
    def tyre_deflection(self, value: float): #tyre_deflection must be > 0.0
        if isinstance(value, float):
            self._tyre_deflection = max(value, self.TOL)
        else:
            raise TypeError

    @property
    def max_width(self) -> float:
        """Maximum width of contact patch (at its center.)"""

    @property
    def max_length(self) -> float:
        """Maximum width of contact patch (at its center.)"""

    @property
    def equation(self) -> callable:
        """Returns callable for computing the width of contact patch depending
        of coordinate along its length.
        """

    @property
    def area(self) -> float:
        """Area of contact patch."""


class RectangleContactPatch(AbstractContactPatch):
    """Rectangle contact patch object that prescribes a rectangular form with
    constant width.

    Child of ``AbstractContactPatch``.

    Attributes
    ----------
    TOL : float
        Tolerance of contact patch parameters calculation (inherited from
        ``AbstractContactPatch``).

    Properties
    ----------
    param : dict
        Dictionary of tyre parameters.
    tyre_deflection : float, optional
        Deflection of a tyre. The default is 0.0.
    max_width : float
        Maximum width of contact patch (at its center).
    max_length : float
        Maximum length of contact patch (at its center).
    equation : callable[x_coord : float] -> float
        Equation for computing the width of contact patch depending
        of coordinate ``x_coord`` along its length.
    area : float
        Area of contact patch.

    Methods
    -------
    None.
    """
    @property
    def max_width(self) -> float:
        """This property overrides :attr:`AbstractContactPatch.max_width`.
        For rectangular contact patch it is assumed that maximum width is
        constant during tyre deflection under loading and equals to width of
        tyre.

        .. warning:: For consistency of tyre parameters dictionary structure,
           curvature radius :math:`R_w` is chosen for the role of tyre width.
        """
        return self.param["forcing"]["R_w"]

    @property
    def max_length(self) -> float:
        r"""This property overrides :attr:`AbstractContactPatch.max_length`.
        Maximum length :math:`l_c` of contact patch (actually, half of total
        length) calculated by

        .. math:: l_c = \sqrt{2 R_t \eta - \eta^2},

        where
            * :math:`R_t` -- radius of tyre tread,
            * :math:`\eta` -- tyre deflection.
            """
        r_t = self.param["forcing"]["R_t"]
        eta = self.tyre_deflection
        return np.sqrt(2 * r_t * eta - eta**2)

    @property
    def equation(self) -> callable:
        """This property overrides :attr:`AbstractContactPatch.equation`.
        For rectangular contact patch it is assumed that maximum width is
        constant during tyre deflection under loading and equals to width of
        tyre."""
        def set_equation(x_coord: float) -> callable: #pylint: disable=W0613
            return self.max_width
        return np.vectorize(set_equation) #returns callable

    @property
    def area(self) -> float:
        """This property overrides :attr:`AbstractContactPatch.area`.
        Area of rectangle. It must be concidered that :attr:`self.max_width` and
        :attr:`self.max_length` are the halves of total width and length of
        rectangle!

        .. math:: A_c = 4 l_c b_c,

        where:
            * :math:`l_c` -- maximum length of contact patch,
            * :math:`b_c` -- maximum width of contact patch.
        """
        return 4 * self.max_width * self.max_length


class EllipseContactPatch(RectangleContactPatch):
    """Ellipse contact patch object that prescribes a ellipse form with
    constant ratio of principal axes.

    Child of ``RectangleContactPatch``.

    Attributes
    ----------
    TOL : float
        Tolerance of contact patch parameters calculation (inherited from
        ``RectangleContactPatch``).

    Properties
    ----------
    param : dict
        Dictionary of tyre parameters.
    tyre_deflection : float, optional
        Deflection of a tyre. The default is 0.0.
    max_width : float
        Maximum width of contact patch (at its center).
    max_length : float
        Maximum length of contact patch (at its center).
    equation : callable[x_coord : float] -> float
        Equation for computing the width of contact patch depending
        of coordinate ``x_coord`` along its length.
    area : float
        Area of contact patch.

    Methods
    -------
    None.
    """
    @property
    def max_width(self) -> float:
        r"""This property overrides :attr:`RectangleContactPatch.max_width`.
        Maximum width :math:`b_c` of contact patch (actually, half of total
        width) calculated by

        .. math:: b_c = \sqrt{2 R_w \eta - \eta^2},

        where
            * :math:`R_w` -- curvature radius of tread in meridian direction,
            * :math:`\eta` -- tyre deflection.
        """
        r_w = self.param["forcing"]["R_w"]
        eta = self.tyre_deflection
        return np.sqrt(2 * r_w * eta - eta**2)

    @property
    def equation(self) -> callable:
        r"""This property overrides :attr:`RectangleContactPatch.equation`.

        Equation of the ellipse:

            .. math:: \frac{x^2}{l_c^2} + \frac{z^2}{b_c^2} = 1,

        so

            .. math:: z = \sqrt{1 - \left(\frac{x}{l_c}\right)^2}.
        """
        def set_equation(x_coord: float) -> callable:
            return (self.max_width *
                    np.sqrt(1 - (x_coord / self.max_length)**2))
        return np.vectorize(set_equation)

    @property
    def area(self) -> float:
        r"""This property overrides :attr:`RectangleContactPatch.area`.

        Area of the ellipse:

            .. math:: A_c = \pi l_c b_c.
        """
        return np.pi * self.max_width * self.max_length


class OvalContactPatch(EllipseContactPatch):
    """Oval contact patch object that prescribes a form of torus and plane
    intersection (Persei curve).

    Child of ``EllipseContactPatch``.

    Attributes
    ----------
    TOL : float
        Tolerance of contact patch parameters calculation.

    Properties
    ----------
    param : dict
        Dictionary of tyre parameters.
    tyre_deflection : float, optional
        Deflection of a tyre. The default is 0.0.
    max_width : float
        Maximum width of contact patch (at its center.)
    max_length : float
        Maximum length of contact patch (at its center.)
    equation : callable[x_coord : float] -> float
        Equation for computing the width of contact patch depending
        of coordinate ``x_coord`` along its length.
    area : float
        Area of contact patch.

    Methods
    -------
    None.
    """
    @property
    def equation(self) -> callable:
        r"""This property overrides :attr:`EllipseContactPatch.equation`.

        Equation for width of Persei curve:

            .. math::
                    z &= \sqrt{-2 R_t^2 + 2 R_t R_w + 2 R_t \eta +
                    2 (R_t - R_w) f(x) - \eta^2 - x^2}, \\
                    f(x) &= {R_t^2 - 2 R_t \eta + \eta^2 + x^2}.
        """
        def set_equation(x_coord: float) -> callable:
            r_w = self.param["forcing"]["R_w"]
            r_t = self.param["forcing"]["R_t"]
            eta = self.tyre_deflection
            fx = np.sqrt(r_t**2 - 2 * r_t * eta + eta**2 + x_coord**2)
            return (np.sqrt(-2 * r_t**2 + 2 * r_t * r_w + 2 * r_t * eta +
                            2 * (r_t - r_w) * fx - eta**2 - x_coord**2))
        return np.vectorize(set_equation)

    @property
    def area(self) -> float:
        r"""This property overrides :attr:`EllipseContactPatch.area`.
        Area under Persei curve can be computed only numerically:

            .. math:: A_c = 4 \int\limits_0^{l_c} z(x) dx,

        where :math:`z(x)` -- equation that represents half of curve.
        Computation of integral is done by :meth:`scipy.integrate.quad` method.
        As a upper bound of integration :math:`l_c - \varepsilon` is used
        instead of simply :math:`l_c` to ensure that there will be no complex
        valued integrand; :math:`\varepsilon` is a very small positive value.
        """
        area, _ = quad(lambda x: 4 * self.equation(x),
                       0.0,
                       self.max_length-self.TOL,
                       epsabs=self.TOL,
                       epsrel=self.TOL,
                       limit=200)
        return area


class TyreSinglePointModel(): # pylint: disable=R0902
    r"""**Tyre model object with single point contact model.**

    Vertical reaction is modeled by V.L.Biderman formula:

        .. math:: F_y = \frac{\eta^2}{C_1 + C_2 \frac{\eta}{p_t}},

        where:
            * :math:`\eta` -- tyre deflection,
            * :math:`C_1` -- model parameter corresponding to tyre tread
              compliance,
            * :math:`C_2` -- model parameter corresponding to tyre structural
              compliance,
            * :math:`p_t` -- tyre air (nitrogen) pressure.

    Attributes
    ----------
    PATCH_TYPES : set
        Allowed types of contact patch.
    REG_COEFF : float
        Regularisation coefficient :math:`\alpha` used for smoothing of
        discontinous functions (:meth:`TyreSinglePointModel.`)
    MIN_HORZ_VEL : float
        Minimal horizontal velocity to avoid singularities during slip ratio
        calculation.

    Properties
    ----------
    deflection : float
        Deflection of the tyre in direction of axis-to-ground projection.
    deflection_rate : float
        Rate of tyre penetration.
    horz_vel : float
        Horizontal velocity of wheel axis.
    ang_vel : float
        Angular velocity of wheel.
    patch_type : str
        Type of contact patch. Allowed values are "rectangle", "ellipse" and
        "oval".
    """
    PATCH_TYPES = {"rectangle", "ellipse", "oval"}
    REG_COEFF = 1000.0
    MIN_HORZ_VEL = 1.0e-8
    TOL = 1e-6
    def __init__(self, json_file, patch_type: str ="ellipse"):
        """
        Constructor of TyreSinglePointModel class.

        Parameters
        ----------
        json_file : str or dict
            Path to JSON file with tyre parameters. #TODO: correct
            String containing the type of contact patch used for enveloping.
            The default is "ellipse".

        Raises
        ------
        ValueError
            If patch_type is not "rectangle", "ellipse" or "oval".

        Returns
        -------
        None.

        """
        self.deflection = 0.0
        self.deflection_rate = 0.0
        self.horz_vel = 0.0
        self.ang_vel = 0.0

        self.invalidate_state()

        if isinstance(json_file, str):
            with open(json_file, "r", encoding="utf-8") as file:
                param = json.load(file)
        else:
            param = json_file

        self.param = param
        self.param_vals = list(self.param["explicit"].values())
        self.patch_type = patch_type

        if self.patch_type == "rectangle":
            self.contact_patch = RectangleContactPatch(self.param)
        elif self.patch_type == "ellipse":
            self.contact_patch = EllipseContactPatch(self.param)
        elif self.patch_type == "oval":
            self.contact_patch = OvalContactPatch(self.param)

    def invalidate_state(self):
        self._vertical_force = None
        self._horizontal_force = None
        self._braking_torque = None
        self._slip_ratio = None
        self._friction_coeff = None

    @property
    def deflection(self):
        return self._deflection
    @deflection.setter
    def deflection(self, value):
        self._deflection = value
        self.invalidate_state()

    @property
    def deflection_rate(self):
        return self._deflection_rate
    @deflection_rate.setter
    def deflection_rate(self, value):
        self._deflection_rate = value
        self.invalidate_state()

    @property
    def horz_vel(self):
        return self._horz_vel
    @horz_vel.setter
    def horz_vel(self, value):
        self._horz_vel = value
        self.invalidate_state()

    @property
    def ang_vel(self):
        return self._ang_vel
    @ang_vel.setter
    def ang_vel(self, value):
        self._ang_vel = value
        self.invalidate_state()

    @property
    def vertical_force(self):
        if self._vertical_force is None:
            self._vertical_force = self.calculate_vertical_force()
        return self._vertical_force

    @property
    def horizontal_force(self):
        if self._horizontal_force is None:
            self._horizontal_force = self.calculate_horizontal_force()
        return self._horizontal_force

    @property
    def braking_torque(self):
        if self._braking_torque is None:
            self._braking_torque = self.calculate_braking_torque()
        return self._braking_torque

    @property
    def slip_ratio(self):
        if self._slip_ratio is None:
            self._slip_ratio = self.calculate_slip_ratio()
        return self._slip_ratio
    @slip_ratio.setter
    def slip_ratio(self, value):
        self._slip_ratio = value

    @property
    def friction_coeff(self):
        if self._friction_coeff is None:
            self._friction_coeff = self.calculate_friction_coeff()
        return self._friction_coeff

    @property
    def patch_type(self) -> str:
        """Type of contact patch. Allowed values are "rectangle", "ellipse" and
        "oval".

        :getter: Returns string of contact type patch name.
        :setter: Sets string of contact type patch name.
        :type: str
        """
        return self._patch_type
    @patch_type.setter
    def patch_type(self, value: str):
        if value in self.PATCH_TYPES:
            self._patch_type = value
        else:
            raise ValueError(
                "Patch type must be 'rectangle', 'ellipse' or 'oval'.")

    def _heaviside(self, x: float) -> float:
        r"""Heaviside step function:

             .. math::

                 \begin{equation}
                 \mathrm{H}(x) =
                 \begin{cases}
                     1,& x>0;\\
                     \frac{1}{2},& x=0;\\
                     -1,& x<0.
                 \end{cases}
                 \end{equation}

        This implementation, however, uses regularized (smooth) version of
        Heaviside step function for the sake of better behaviour during numeric
        integration. There are many regularized versions of Heaviside step
        function, so it is the chosen one:

            .. math:: \mathrm{H}(x) = \frac{1}{2} \tanh(\alpha x) + \frac{1}{2},

        where :math:`\alpha` -- coefficient of regularisation. More bigger
        :math:`\alpha` is, more closer smoothed version of function approaches
        discontinous original function.
        """
        return 0.5 * np.tanh(self.REG_COEFF * x) + 0.5

    def _sign(self, x: float) -> float:
        r"""Sign function:

        .. math::

            \begin{equation}
            \mathrm{sign}(x) =
            \begin{cases}
                1,& x>0;\\
                0,& x=0;\\
                -1,& x<0.
            \end{cases}
            \end{equation}

        This implementation, however, uses regularized (smooth) version of
        sign function for the sake of better behaviour during numeric
        integration. There are many regularized versions of sign function,
        so it is the chosen one:

            .. math:: \mathrm{sign}(x) = \tanh(\alpha x),

        where :math:`\alpha` -- coefficient of regularisation. More bigger
        :math:`\alpha` is, more closer smoothed version of function approaches
        discontinous original function.
         """
        return np.tanh(self.REG_COEFF * x)

    def calculate_deflection(self,
                             distance_to_axis: float,
                             velocity_to_axis: float) -> float:
        r"""Calculate deflection of the tyre and its rate of penetration under
        the axis from formula:

        .. math:: \eta &= (R_t - y_1) \mathrm{H} (R_t - y_1), \\
           \dot{\eta} &= - \dot{y}_1 \mathrm{H} (R_t - y_1),

        where:
            * :math:`y_1` -- distance to axis,
            * :math:`R_t` -- radius of tyre tread,
            * :math:`\mathrm{H}` -- Heaviside step function.
        """
        eta_unconstr = self.param["forcing"]["R_t"] - distance_to_axis
        self.deflection = eta_unconstr * self._heaviside(eta_unconstr)
        self.deflection_rate = (- velocity_to_axis *
                                     self._heaviside(eta_unconstr))

    def calculate_vertical_stiffness(self) -> float:
        r"""Calculate vertical stiffness of tyre at given tyre deflection value.
        According to V.L. Biderman model, vertical (radial) stiffness of tyre
        at given tyre displacement can be computed by this formula:

            .. math:: k_t = \frac{2 C_1 + C_2 \frac{\eta}{p_t}}
               {\left(C1 + C_2 \frac{\eta}{p_t}\right)^2} \eta,

         where:
             * :math:`\eta` -- tyre deflection,
             * :math:`C_1` -- model parameter corresponding to tyre tread
               compliance,
             * :math:`C_2` -- model parameter corresponding to tyre structural
               compliance,
             * :math:`p_t` -- tyre air (nitrogen) pressure.

        Parameters
        ----------
        None.

        Returns
        -------
        stiffness : float
            Tyre vertical stiffness.

        """
        stiffness = (self.deflection *
                     self.param["forcing"]['pt'] *
                    (2 * self.param["forcing"]['C1'] *
                     self.param["forcing"]['pt'] +
                     self.param["forcing"]['C2'] * self.deflection) /
                    (self.param["forcing"]['C1'] *
                     self.param["forcing"]['pt'] +
                     self.param["forcing"]['C2'] * self.deflection)**2)

        if isinstance(stiffness, np.ndarray):
            stiffness[stiffness < 0.0] = 0.0
        elif isinstance(stiffness, float):
            stiffness = max(stiffness, 0.0)
        return stiffness

    def calculate_nat_freq(self) -> float:
        r"""Calculate fundamental natural frequency of tyre vibration. Wheel
        with tyre interpreted as single DOF system as a point mass on to-ground
        spring:

            .. math:: \omega_0 = \sqrt{\frac{k_t}{m_1}},

        where:
            * :math:`k_t` -- tyre vertical stiffness,
            * :math:`m_1` -- mass of the wheel.

        Parameters
        ----------
        None.

        Returns
        -------
        natural_frequency : float
            Fundamental frequency of tyre vertical vibration.

        """
        try:
            natural_frequency = np.sqrt(self.calculate_vertical_stiffness() /
                                        self.param["explicit"]["m1"])
        except FloatingPointError:
            print(self.calculate_vertical_stiffness())
            natural_frequency = 0.0
        return natural_frequency

    def calculate_vertical_force(self) -> float:
        r"""Calculate total vertical dynamic reaction of tyre as a sum of two
        components:

            .. math:: F_y =
               \begin{cases}
                   F_y^{el} + F_y^{in}, \quad (F_y^{el}>0) \cap (F_y^{in}>0), \\
                   0 \quad \text{otherwise}.
               \end{cases}

        Restoring vertical reaction is modeled by V.L.Biderman formula:

            .. math:: F_y^{el} = \frac{\eta^2}{C_1 + C_2 \frac{\eta}{p_t}},

        where:
            * :math:`\eta` -- tyre deflection,
            * :math:`C_1` -- model parameter corresponding to tyre tread
              compliance,
            * :math:`C_2` -- model parameter corresponding to tyre structural
              compliance,
            * :math:`p_t` -- tyre air (nitrogen) pressure.

        Aside restoring force, there is a damping force, considered as viscous:

            .. math:: F_y^{in} = 2 \zeta m_1 \omega_0 \dot{\eta},

        where:
            * :math:`\zeta` -- critical damping ratio,
            * :math:`m_1` -- mass of the wheel,
            * :math:`\omega_0` -- fundamental frequency of tyre vertical vibration,
            * :math:`\dot{\eta}` -- rate of tyre deflection value (velocity of
              axis with negative sign):

            .. math:: \eta = (R_t - y_1) \mathrm{H}(R_t - y_1) \to \underline{
                \dot{\eta} = -\dot{y_1} \mathrm{H}(R_t - y_1)}.


        Parameters
        ----------
        None.

        Returns
        -------
        total_force : float
            Total vertical force.

        """
        restoring_force = (self.deflection**2 /
                           (self.param["forcing"]["C1"] +
                            self.param["forcing"]["C2"] *
                            self.deflection / self.param["forcing"]["pt"]))
        damping_force = 2 * (self.param["forcing"]["zeta"] *
                             self.param["explicit"]["m1"] *
                             self.calculate_nat_freq() *
                             self.deflection_rate)
        if isinstance(self.deflection, np.ndarray):
            restoring_force[restoring_force < 0.0] = 0.0
            damping_force[damping_force < 0.0] = 0.0
        elif isinstance(self.deflection, float):
            restoring_force = max(restoring_force, 0.0)
            damping_force = max(damping_force, 0.0)
        total_force = restoring_force + damping_force
        return total_force

    def calculate_slip_ratio(self) -> float:
        if self.deflection < self.TOL or self.ang_vel < self.TOL:
            return 0.0
        slip_ratio = (1 - self.ang_vel *
                          self.param["forcing"]["R_t"] / (self.horz_vel +
                                                          self.MIN_HORZ_VEL))
        return slip_ratio

    def calculate_slip_ratio_diff(self, d2vhdt2: float) -> float:
        self.contact_patch.tyre_deflection = self.deflection
        if self.deflection <= self.TOL or self.ang_vel < self.TOL:
            return 0.0
        coeff = -np.abs(self.horz_vel) / (self.contact_patch.max_length + 1)
        rhs = (-np.abs(self.horz_vel) + self._sign(d2vhdt2) * self.ang_vel *
               self.param["forcing"]["R_t"]) / self.contact_patch.max_length
        return coeff * self.slip_ratio + rhs

    def calculate_friction_coeff(self) -> float:
        mu_slip = (self.param["forcing"]["mu0"] *
                   np.tanh(self.param["forcing"]["a1"] * self.slip_ratio))
        return mu_slip

    def calculate_horizontal_force(self) -> float:
        horizontal_force = self.friction_coeff * self.vertical_force
        return horizontal_force

    def calculate_braking_torque(self) -> float:
        braking_torque = (self.horizontal_force *
                          (self.param["forcing"]["R_t"] - self.deflection))
        return braking_torque


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(['science', 'ieee', 'russian-font'])

    JSON_FILE = ("C:/Users/devoi/Thesis/dev/"
                        "aircraft-taxiing-vibrations/parameters_data/"
                        "MLG_tyre_properties.json")
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        param_dict = json.load(f)

    rcp = RectangleContactPatch(param_dict, 50.0)
    ecp = EllipseContactPatch(param_dict, 50.0)
    ocp = OvalContactPatch(param_dict, 50.0)
    deflections = np.linspace(0, 200.0, 11)
    for deflection in deflections:
        for patch in (rcp, ecp, ocp):
            patch.tyre_deflection = deflection
        print("--------------------------------------------------------------")
        print(f"At tyre deflecton = {deflection:.2f} mm:")
        print(f"Rectangle area: {rcp.area:.2f}")
        print(f"Ellipse area: {ecp.area:.2f}")
        print(f"Oval area: {ocp.area:.2f}")
        print(f"Ellipse area rel err: "
              f"{(ecp.area - ocp.area) / ocp.area * 100:.2f}%")
        print(f"Rectangle area rel err: "
              f"{(rcp.area - ocp.area) / ocp.area * 100:.2f}%")
        print("\n")

    tyre = TyreSinglePointModel(JSON_FILE)
    tyre.deflection = np.linspace(0.0, 150.0, 100)
    tyre.slip_ratio = np.linspace(-1.0, 1.0, 100)

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(4,1.7), dpi=200)

    ax1.plot(tyre.deflection, 1e-3 * tyre.vertical_force)
    ax1.set(xlabel=r"$\eta$, мм", ylabel=r"$F_y$, кН")
    ax1.grid(True, which="both")
    ax1.margins(0.0, 0.05)

    ax2.plot(tyre.slip_ratio, tyre.friction_coeff)
    ax2.set(xlabel=r"$s$", ylabel=r"$\mu_{slip}$")
    ax2.grid(True, which="both")
    ax2.margins(0.0, 0.05)

    fig.tight_layout()
    plt.show()
