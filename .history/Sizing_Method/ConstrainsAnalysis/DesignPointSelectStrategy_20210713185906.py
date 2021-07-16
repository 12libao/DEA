# author: Bao Li #
# Georgia Institute of Technology #
import sys
import os

sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib.pylab as plt
import Sizing_Method.Other.US_Standard_Atmosphere_1976 as atm
import Sizing_Method.Aerodynamics.ThrustLapse as thrust_lapse
import Sizing_Method.Aerodynamics.Aerodynamics as ad
import Sizing_Method.ConstrainsAnalysis.ConstrainsAnalysis as ca
import Sizing_Method.ConstrainsAnalysis.ConstrainsAnalysisPD as ca_pd
import Sizing_Method.ConstrainsAnalysis.ConstrainsAnalysisPDP1P2 as ca_pd_12
from scipy.optimize import curve_fit


"""
The unit use is IS standard
"""


class Design_Point_Select_S:
    """This is a power-based master constraints analysis"""

    def __init__(self, altitude, velocity, beta, wing_load, Hp=0.2, number_of_motor=12, C_DR=0):
        """

        :param beta: weight fraction
        :param Hp: P_motor/P_total
        :param n: number of motor
        :param K1: drag polar coefficient for 2nd order term
        :param K2: drag polar coefficient for 1st order term
        :param C_D0: the drag coefficient at zero lift
        :param C_DR: additional drag caused, for example, by external stores,
        braking parachutes or flaps, or temporary external hardware

        :return:
            power load: P_WTO
        """

        self.h = altitude
        self.v = velocity
        self.rho = atm.atmosphere(geometric_altitude=self.h).density()

        self.beta = beta
        self.hp = Hp
        self.n = number_of_motor

        # power lapse ratio
        self.alpha = thrust_lapse.thrust_lapse_calculation(altitude=self.h,
                                                           velocity=self.v).high_bypass_ratio_turbofan()

        self.k1 = ad.aerodynamics_without_pd(self.h, self.v).K1()
        self.k2 = ad.aerodynamics_without_pd(self.h, self.v).K2()
        self.cd0 = ad.aerodynamics_without_pd(self.h, self.v).CD_0()
        self.cdr = C_DR

        self.w_s = wing_load
        self.g0 = 9.80665

        self.coefficient = (1 - self.hp) * self.beta * self.v / self.alpha

        # Estimation of ΔCL and ΔCD
        pd = ad.aerodynamics_with_pd(
            self.h, self.v, Hp=self.hp, n=self.n, W_S=self.w_s)
        self.q = 0.5 * self.rho * self.v ** 2
        self.cl = self.beta * self.w_s / self.q
        # print(self.cl)
        self.delta_cl = pd.delta_lift_coefficient(self.cl)
        self.delta_cd0 = pd.delta_CD_0()

    def master_equation(self, n, dh_dt, dV_dt):
        cl = self.cl * n + self.delta_cl

        cd = self.k1 * cl ** 2 + self.k2 * cl + self.cd0 + self.cdr + self.delta_cd0
        p_w = self.coefficient * \
            (self.q / (self.beta * self.w_s) *
             cd + dh_dt / self.v + dV_dt / self.g0)
        return p_w
