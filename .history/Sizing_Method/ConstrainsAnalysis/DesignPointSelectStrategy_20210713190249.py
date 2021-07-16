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


class Design_Point_Select_Strategy:
    """This is a design point select strategy from constrains analysis"""

    def __init__(self, altitude, velocity, beta, Hp, number_of_motor=12, C_DR=0):
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
            power load: design point p/w and w/s
        """

        self.h = altitude
        self.v = velocity
        self.rho = atm.atmosphere(geometric_altitude=self.h).density()

    def master_equation(self, n, dh_dt, dV_dt):
        cl = self.cl * n + self.delta_cl

        cd = self.k1 * cl ** 2 + self.k2 * cl + self.cd0 + self.cdr + self.delta_cd0
        p_w = self.coefficient * \
            (self.q / (self.beta * self.w_s) *
             cd + dh_dt / self.v + dV_dt / self.g0)
        return p_w
