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

    def __init__(self, altitude, velocity, beta, p_turbofan_max, p_motorfun_max):
        """
        :param altitude: m x 1 matrix
        :param velocity: m x 1 matrix
        :param beta: P_motor/P_total m x 1 matrix
        :param p_turbofan_max: maximum 

        the first group of condition is for stall speed

        :return:
            power load: design point p/w and w/s
        """

        self.h = altitude
        self.v = velocity
        self.beta = beta

        # initialize the p_w, w_s, hp, n, m
        self.n = 100
        self.m = len(self.h)
        boundary_lower = 5000
        boundary_upper = 7000

        self.p_w = np.zeros([self.m, self.n])  # m x n matrix
        self.w_s = np.linspace(boundary_lower, boundary_upper, self.n)
        self.hp = np.linspace(0, 1, 100)

    def strategy(self):
        
        methods = [ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun,
                ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun,
                ConstrainsAnalysis_Mattingly_Method_with_DP_electric,
                ConstrainsAnalysis_Gudmundsson_Method_with_DP_electric,
                ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun,
                ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun,
                ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun,
                ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun]
        subtitle = ["0", "1", '2', '3', '4', '5', '6', '7']

        



        cl = self.cl * n + self.delta_cl

        cd = self.k1 * cl ** 2 + self.k2 * cl + self.cd0 + self.cdr + self.delta_cd0
        p_w = self.coefficient * \
            (self.q / (self.beta * self.w_s) *
             cd + dh_dt / self.v + dV_dt / self.g0)
        return p_w
