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
from icecream import ic 


"""
The unit use is IS standard
"""


class Design_Point_Select_Strategy:
    """This is a design point select strategy from constrains analysis"""

    def __init__(self, altitude, velocity, beta, method, p_w_turbofan_max=72, p_w_motorfun_max=72, n=12):
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
        self.p_w_turbofan_max = p_w_turbofan_max
        self.p_w_motorfun_max = p_w_motorfun_max

        # initialize the p_w, w_s, hp, n, m
        self.n = 100
        self.m = altitude.size

        self.hp = np.linspace(0, 1, self.n)
        self.hp_threshold = self.p_w_motorfun_max / (self.p_w_motorfun_max + self.p_w_turbofan_max)

        # method1 = Mattingly_Method, method2 = Gudmundsson_Method
        if method == 1:
            self.method1 = ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun
            self.method2 = ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_electric
        else:
            self.method1 = ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun
            self.method2 = ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_electric

        problem = self.method1(self.h[0], self.v[0], self.beta[0], 6000, self.hp_threshold)
        self.w_s = problem.allFuncs[0](problem)
        
    def p_w_compute(self):
        p_w = np.zeros([self.m, self.n])  # m x n matrix
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
                
                if p_w_1 > self.p_w_turbofan_max:
                        p_w_1 = 100000
                elif p_w_2 > self.p_w_motorfun_max:
                        p_w_2 = 100000
                
                p_w[i, j] = p_w_1 + p_w_2

        return p_w

    def strategy(self):
        p_w = Design_Point_Select_Strategy.p_w_compute(self)

        #find the min p_w for difference hp for each flight condition:
        p_w_min = np.amin(p_w, axis=1)
        ic

        #find the index of p_w_min which is the hp
        hp_p_w_min = np.array(np.where(p_w == p_w_min))

        #find the max p_w_min for each flight condition which is the design point we need:
        design_point = np.array([self.w_s, np.amax(p_w_min)])
        return hp_p_w_min, design_point


if __name__ == "__main__":
    constrains = np.array([[0, 80, 1, 0.2], [0, 68, 0.988, 0.5],  [11300, 230, 0.948, 0.8],
                           [11900, 230, 0.78, 0.8], [3000, 100,
                                                     0.984, 0.8], [0, 100, 0.984, 0.5],
                           [3000, 200, 0.975, 0.6], [7000, 230, 0.96, 0.7]])
    h = constrains[:, 0]
    v = constrains[:, 1]
    beta = constrains[:, 2]

    problem = Design_Point_Select_Strategy(h, v, beta, method=2)
    hp_p_w_min, design_point = problem.strategy()
    ic(hp_p_w_min, design_point)
    
