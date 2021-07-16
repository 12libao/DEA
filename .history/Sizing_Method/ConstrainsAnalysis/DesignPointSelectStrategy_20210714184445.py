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

    def __init__(self, altitude, velocity, beta, method, p_turbofan_max, p_motorfun_max, n=12):
        """
        :param altitude: m x 1 matrix
        :param velocity: m x 1 matrix
        :param beta: P_motor/P_total m x 1 matrix
        :param p_turbofan_max: maximum propulsion power for turbofan (threshold value)
        :param p_motorfun_max: maximum propulsion power for motorfun (threshold value)
        :param n: number of motor

        the first group of condition is for stall speed
        the stall speed condition have to use motor, therefore with PD

        :return:
            power load: design point p/w and w/s
        """

        self.h = altitude
        self.v = velocity
        self.beta = beta
        self.n_motor = n
        self.p_turbofan_max = p_turbofan_max
        self.p_motorfun_max = p_motorfun_max

        # initialize the p_w, w_s, hp, n, m
        self.n = 100
        self.m = len(self.h)

        self.hp = np.linspace(0, 1, self.n)
        self.hp_threshold = self.p_motorfun_max / (self.p_motorfun_max + self.p_turbofan_max)

        # method1 = Mattingly_Method, method2 = Gudmundsson_Method
        if method == 1:
            self.method1 = ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun
            self.method2 = ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_electric
        else:
            self.method1 = ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun
            self.method2 = ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_electric

        problem = self.method(self.h[0], self.v[0], self.beta[0], 6000, self.hp_threshold)
        self.w_s = problem.allFuncs[0](problem)
        
    def p_w_compute(self):
        self.p_w = np.zeros([self.m, self.n])  # m x n matrix
        for i in range(1, 8):
            for j in range(self.n):
                problem1 = self.method1(self.h[i], self.v[i],
                                    self.beta[i], self.w_s, self.hp[j])
                problem2 = self.method2(self.h[i], self.v[i],
                                        self.beta[i], self.w_s, self.hp[j])
                if i >= 5:
                    p_w_1 = problem1.allFuncs[-1](problem1, roc=15 - 5 * (i - 5))
                    p_w_2 = problem2.allFuncs[-1](problem2, roc=15 - 5 * (i - 5))
                else:
                    p_w_1 = problem1.allFuncs[i](problem1)
                    p_w_2 = problem2.allFuncs[i](problem2)
                
                if p_w_1 > self.p_turbofan_max:
                        p_w_1 = 100000
                elif p_w_2 > self.p_motorfun_max:
                        p_w_2 = 100000
                
                self.p_w[i, j] = p_w_1 + p_w_2

        return self.p_w

    def strategy(self):
        #find the min p_w for difference hp for each flight condition:
        for i in 

