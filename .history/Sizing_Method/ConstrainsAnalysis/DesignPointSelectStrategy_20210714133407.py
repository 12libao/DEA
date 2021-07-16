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

    def __init__(self, altitude, velocity, beta, p_turbofan_max, p_motorfun_max, n=12):
        """
        :param altitude: m x 1 matrix
        :param velocity: m x 1 matrix
        :param beta: P_motor/P_total m x 1 matrix
        :param p_turbofan_max: maximum propulsion power for turbofan (threshold value)
        :param p_motorfun_max: maximum propulsion power for motorfun (threshold value)
        :param n: number of motor

        the first group of condition is for stall speed

        :return:
            power load: design point p/w and w/s
        """

        self.h = altitude
        self.v = velocity
        self.beta = beta
        sem
        self.p_turbofan_max = p_turbofan_max
        self.p_motorfun_max = p_motorfun_max

        # initialize the p_w, w_s, hp, n, m
        self.n = 100
        self.m = len(self.h)

        boundary_lower = 5000
        boundary_upper = 7000
        self.p_w = np.zeros([self.m, self.n])  # m x n matrix
        self.w_s = np.linspace(boundary_lower, boundary_upper, self.n)

        hp_max = 
        self.hp = np.linspace(0, 1, 100)

    def strategy(self):
        
        
        return p_w
