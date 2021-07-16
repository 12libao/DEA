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
import math


"""
The unit use is IS standard
"""


class Design_Point_Select_Strategy:
    """This is a design point select strategy from constrains analysis"""

    def __init__(self, altitude, velocity, beta, method=2, strategy_apply=0, propulsion_constrains=0, n=12):
        """
        :param altitude: m x 1 matrix
        :param velocity: m x 1 matrix
        :param beta: P_motor/P_total m x 1 matrix
        :param p_turbofan_max: maximum propulsion power for turbofan (threshold value)
        :param p_motorfun_max: maximum propulsion power for motorfun (threshold value)
        :param n: number of motor

        :param method: if method = 1, it is Mattingly Method, otherwise is Gudmundsson Method
        :param strategy_apply: if strategy_apply = 0, no strategy apply
        :param propulsion_constrains: if propulsion_constrains = 0, no propulsion_constrains apply

        the first group of condition is for stall speed
        the stall speed condition have to use motor, therefore with PD

        :return:
            power load: design point p/w and w/s
        """

        self.h = altitude
        self.v = velocity
        self.beta = beta
        self.n_motor = n
        self.propulsion_constrains = propulsion_constrains
        self.strategy_apply = strategy_apply

        # initialize the p_w, w_s, hp, n, m
        self.n = 100
        self.m = altitude.size

        self.hp = np.linspace(0, 1, self.n+1)
        self.hp_threshold = 0.5

        # method = 1 = Mattingly_Method, method = 2 = Gudmundsson_Method
        if method == 1:
            self.method1 = ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun
            self.method2 = ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_electric
        else:
            self.method1 = ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun
            self.method2 = ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_electric

        problem = self.method1(
            self.h[0], self.v[0], self.beta[0], 6000, self.hp_threshold)
        self.w_s = problem.allFuncs[0](problem)

    def p_w_compute(self, p_w_turbofan_max, p_w_motorfun_max, pc):
        p_w = np.zeros([self.m, len(self.hp)])  # m x (n+1) matrix
        p_w_1 = np.zeros([self.m, len(self.hp)])  # m x (n+1) matrix
        p_w_2 = np.zeros([self.m, len(self.hp)])  # m x (n+1) matrix
        for i in range(1, 8):
            for j in range(len(self.hp)):
                problem1 = self.method1(self.h[i], self.v[i],
                                    self.beta[i], self.w_s, self.hp[j])
                problem2 = self.method2(self.h[i], self.v[i],
                                        self.beta[i], self.w_s, self.hp[j])
                if i >= 5:
                    p_w_1[i, j] = problem1.allFuncs[-1](problem1, roc=15 - 5 * (i - 5))
                    p_w_2[i, j] = problem2.allFuncs[-1](problem2, roc=15 - 5 * (i - 5))
                else:
                    p_w_1[i, j] = problem1.allFuncs[i](problem1)
                    p_w_2[i, j] = problem2.allFuncs[i](problem2)
                
                if self.propulsion_constrains != 0 and pc != 0:
                    if p_w_1[i, j] > p_w_turbofan_max:
                        p_w_1[i, j] = 100000
                    elif p_w_2[i, j] > p_w_motorfun_max:
                        p_w_2[i, j] = 100000
                    
                p_w[i, j] = p_w_1[i, j] + p_w_2[i, j]

        return p_w, p_w_1, p_w_2

    def p_w_min(self, p_w):
        #find the min p_w for difference hp for each flight condition:
        p_w_min = np.amin(p_w, axis=1)

        #find the index of p_w_min which is the hp
        hp_p_w_min = np.zeros(8)
        for i in range(1, 8):
            for j in range(len(self.hp)):
                if p_w[i, j] - p_w_min[i] < 0.001:
                    hp_p_w_min[i] = j * 0.01

        p_w_1 = np.zeros(8)
        p_w_2= np.zeros(8)
        for i in range(1, 8):
            problem1 = self.method1(
                self.h[i], self.v[i], self.beta[i], self.w_s, hp_p_w_min[i])
            problem2 = self.method2(
                self.h[i], self.v[i], self.beta[i], self.w_s, hp_p_w_min[i])
            if i >= 5:
                p_w_1[i] = problem1.allFuncs[-1](
                    problem1, roc=15 - 5 * (i - 5))
                p_w_2[i] = problem2.allFuncs[-1](
                    problem2, roc=15 - 5 * (i - 5))
            else:
                p_w_1[i] = problem1.allFuncs[i](problem1)
                p_w_2[i] = problem2.allFuncs[i](problem2)

        p_w_min = np.amax(p_w_min)
        p_w_1_min = np.amax(p_w_1)
        p_w_2_min = np.amax(p_w_2)
        return p_w_1_min, p_w_2_min, p_w_min, hp_p_w_min

    def strategy(self):
        if self.strategy_apply == 0:
            p_w_turbofan_max = 10000
            p_w_motorfun_max = 10000
            p_w, p_w_1, p_w_2 = Design_Point_Select_Strategy.p_w_compute(
                self, p_w_turbofan_max, p_w_motorfun_max, pc=0)
            p_w_min = np.amax(p_w[:, 50])
            p_w_1_min = np.amax(p_w_1[:, 50])
            p_w_2_min = np.amax(p_w_2[:, 50])
            hp_p_w_min = 0.5*np.ones(8)
        else:
            if self.propulsion_constrains == 0:
                p_w_turbofan_max = 100000
                p_w_motorfun_max = 100000
                p_w, p_w_1, p_w_2 = Design_Point_Select_Strategy.p_w_compute(self, p_w_turbofan_max, p_w_motorfun_max, 0)
            else:
                p_w, _, _ = Design_Point_Select_Strategy.p_w_compute(self, 10000, 10000, pc=0)
                p_w_1_min, p_w_2_min, _, _ = Design_Point_Select_Strategy.p_w_min(self, p_w)
                p_w_turbofun_boundary = math.ceil(p_w_1_min)
                p_w_motorfun_boundary = math.ceil(p_w_2_min)

                ic(p_w_turbofun_boundary, p_w_motorfun_boundary)
                # build p_w_design_point matrix, try p_w_max to find the best one
                p_w_design_point = np.zeros([p_w_turbofun_boundary+1, p_w_motorfun_boundary+1])
                for i in range(p_w_turbofun_boundary+1):
                    for j in range(p_w_motorfun_boundary+1):
                        p_w, _, _ = Design_Point_Select_Strategy.p_w_compute(self, i, j, 1)
                        
                        #find the min p_w from hp: 0 --- 100 for each flight condition:
                        p_w_min = np.amin(p_w, axis=1)
                        p_w_design_point[i, j] = np.amax(p_w_min)
                    print(i)

                p_w_turbofan_max = np.unravel_index(
                    p_w_design_point.argmin(), p_w_design_point.shape)[0]
                p_w_motorfun_max = np.unravel_index(
                    p_w_design_point.argmin(), p_w_design_point.shape)[1]
                p_w, p_w_1, p_w_2 = Design_Point_Select_Strategy.p_w_compute(
                    self, p_w_turbofan_max, p_w_motorfun_max, 1)
            
            # ic(p_w, p_w_1, p_w_2)

            p_w_1_min, p_w_2_min, p_w_min, hp_p_w_min = Design_Point_Select_Strategy.p_w_min(self, p_w)
            hp_p_w_min[0] = p_w_motorfun_max/(p_w_motorfun_max+p_w_turbofan_max)

        return self.w_s, p_w_min, p_w_1_min, p_w_2_min, hp_p_w_min, p_w_turbofan_max, p_w_motorfun_max


if __name__ == "__main__":
    n, m = 250, 8
    w_s = np.linspace(100, 9000, n)
    p_w = np.zeros([m, n, 6])

    constrains = np.array([[0, 80, 1, 0.2], [0, 68, 0.988, 0.5],  [11300, 230, 0.948, 0.8],
                           [11900, 230, 0.78, 0.8], [3000, 100,
                                                     0.984, 0.8], [0, 100, 0.984, 0.5],
                           [3000, 200, 0.975, 0.6], [7000, 230, 0.96, 0.7]])
    constrains_name = ['stall speed', 'take off', 'cruise', 'service ceiling', 'level turn @3000m',
                       'climb @S-L', 'climb @3000m', 'climb @7000m', 'feasible region-hybrid', 'feasible region-conventional']
    l1 = ['hp: 0.5', 'hp: adjust', 'hp: adjust with threshold']
    l2 = ['design point hp: 0.5', 'design point hp: adjust', 'design point hp: adjust with threshold']
    color = ['k', 'c',  'b', 'g', 'y', 'plum', 'violet', 'm']
    color2 = ['r', 'g', 'y']
    l_style = ['-', '--', '-.']
    mark = ['s', '^', '*']
    alpha = [0.4, 0.2, 0.1]
    methods = [ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun,
               ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun,
               ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_electric,
               ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_electric]
    strategy = [0, 1, 1]
    propulsion = [0, 0, 0]

    # plots
    fig, ax = plt.subplots(3, 2, sharey=True, sharex=True, figsize=(10, 10))
    ax = ax.flatten()

    design_point_p_w, design_point_w_s = np.zeros([3, 6]), np.zeros([3, 2])
    for z in range(3):
        h = constrains[:, 0]
        v = constrains[:, 1]
        beta = constrains[:, 2]
        
        problem1 = Design_Point_Select_Strategy(
            h, v, beta, method=1, strategy_apply=strategy[z], propulsion_constrains=propulsion[z])
        problem2 = Design_Point_Select_Strategy(
            h, v, beta, method=2, strategy_apply=strategy[z], propulsion_constrains=propulsion[z])
        
        w_s1, p_w_min11, p_w_1_min11, p_w_2_min11, hp_p_w_min11, p_w_turbofan_max11, p_w_motorfun_max11 = problem1.strategy()
        ic(w_s1, p_w_min11, p_w_1_min11, p_w_2_min11, hp_p_w_min11, p_w_turbofan_max11, p_w_motorfun_max11 )

        design_point_w_s[z, 0], design_point_p_w[z, 4], design_point_p_w[z, 0], design_point_p_w[z, 2], hp_p_w_min_1, _, _ = problem1.strategy()
        design_point_w_s[z, 1], design_point_p_w[z, 5], design_point_p_w[z, 1], design_point_p_w[z, 3], hp_p_w_min_2, _, _ = problem2.strategy()

        for k in range(6):
            for i in range(m):
                for j in range(n):
                    h = constrains[i, 0]
                    v = constrains[i, 1]
                    beta = constrains[i, 2]

                    if k % 2 == 0:
                        hp = hp_p_w_min_1[i]
                    else:
                        hp = hp_p_w_min_2[i]

                    # calculate p_w
                    if k < 4:
                        problem = methods[k](h, v, beta, w_s[j], hp)
                        if i >= 5:
                            p_w[i, j, k] = problem.allFuncs[-1](problem, roc=15 - 5 * (i - 5))
                        else:
                            p_w[i, j, k] = problem.allFuncs[i](problem)
                    else:
                        if i == 0:
                            problem = methods[k-2](h, v, beta, w_s[j], hp)
                            p_w[i, j, k] = problem.allFuncs[i](problem)
                        else:
                            p_w[i, j, k] = p_w[i, j, k-4] + p_w[i, j, k-2]

                # plot the lines
                if z == 0:
                    if i == 0:
                        ax[k].plot(p_w[i, :, k], np.linspace(0, 100, n), 
                                linewidth=1, alpha=0.1, linestyle=l_style[z], label=constrains_name[i])
                    else:
                        ax[k].plot(w_s, p_w[i, :, k], color=color[i],
                                linewidth=1, alpha=0.1, linestyle=l_style[z], label=constrains_name[i])
                else:
                    if i == 0:
                        ax[k].plot(p_w[i, :, k], np.linspace(0, 100, n),
                                   linewidth=1, alpha=0.1, linestyle=l_style[z])
                    else:
                        ax[k].plot(w_s, p_w[i, :, k], color=color[i],
                                   linewidth=1, alpha=0.1, linestyle=l_style[z])

            # plot fill region
            p_w[0, :, k] = 10 ** 10 * (w_s - p_w[0, 0, k])
            ax[k].fill_between(w_s, np.amax(p_w[0:m, :, k], axis=0), 150,  alpha=alpha[z], label=l1[z])
            ax[k].plot(design_point_w_s[z, 0], design_point_p_w[z, k], marker=mark[z],
                       markersize=5, label=l2[z])

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.125, 0.02, 0.75, 0.25), loc="lower left",
               mode="expand", borderaxespad=0, ncol=4, frameon=False)
    hp = constrains[:, 3]
    plt.xlim(200, 9000)
    plt.ylim(0, 100)
    plt.setp(ax[0].set_title(r'$\bf{Mattingly-Method}$'))
    plt.setp(ax[1].set_title(r'$\bf{Gudmundsson-Method}$'))
    plt.setp(ax[4:6], xlabel='Wing Load: $W_{TO}$/S (N/${m^2}$)')
    plt.setp(ax[0], ylabel=r'$\bf{Turbofun}$''\n $P_{SL}$/$W_{TO}$ (W/N)')
    plt.setp(ax[2], ylabel=r'$\bf{Motor}$ ''\n $P_{SL}$/$W_{TO}$ (W/N)')
    plt.setp(
        ax[4], ylabel=r'$\bf{Turbofun+Motor}$' '\n' r'$\bf{vs.Conventional}$ ''\n $P_{SL}$/$W_{TO}$ (W/N)')
    plt.subplots_adjust(bottom=0.15)
    plt.suptitle(r'$\bf{Component}$' ' ' r'$\bf{P_{SL}/W_{TO}}$' ' ' r'$\bf{Diagrams}$'
                 ' ' r'$\bf{After}$' ' ' r'$\bf{Adjust}$' ' ' r'$\bf{Degree-of-Hybridization}$'
                 '\n hp: take-off=' +
                 str(hp[0]) + '  stall-speed=' +
                 str(hp[1]) + '  cruise=' +
                 str(hp[2]) + '  service-ceiling=' +
                 str(hp[3]) + '\n level-turn=@3000m' +
                 str(hp[4]) + '  climb@S-L=' +
                 str(hp[5]) + '  climb@3000m=' +
                 str(hp[6]) + '  climb@7000m=' + str(hp[7]))
    plt.show()
  
