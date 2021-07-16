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

    def __init__(self, altitude, velocity, beta, method, p_w_turbofan_max=20, p_w_motorfun_max=25, n=12):
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

        self.hp = np.linspace(0, 1+1/self.n, self.n+1)
        self.hp_threshold = self.p_w_motorfun_max / (self.p_w_motorfun_max + self.p_w_turbofan_max)

        # method1 = Mattingly_Method, method2 = Gudmundsson_Method
        if method == 1:
            self.method1 = ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun
            self.method2 = ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_electric
        else:
            self.method1 = ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun
            self.method2 = ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_electric

        problem = self.method1(
            self.h[0], self.v[0], self.beta[0], 6000, self.hp_threshold)
        self.w_s = problem.allFuncs[0](problem)

    def p_w_compute(self):
        p_w = np.zeros([self.m, len(self.hp)])  # m x (n+1) matrix
        for i in range(1, 8):
            for j in range(len(self.hp)):
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
        ic(p_w_min)

        #find the index of p_w_min which is the hp
        hp_p_w_min = np.zeros(8)
        for i in range(1, 8):
            for j in range(len(self.hp)):
                if p_w[i, j] - p_w_min[i] < 0.001:
                    hp_p_w_min[i] = j * 0.01

        hp_p_w_min[0] = self.hp_threshold

        #find the max p_w_min for each flight condition which is the design point we need:
        design_point = np.array([self.w_s, np.amax(p_w_min)])
        return hp_p_w_min, design_point

    def no_strategy(self):
        p_w = Design_Point_Select_Strategy.p_w_compute(self)

        #find the min p_w for difference hp for each flight condition:
        p_w_min = np.amin(p_w, axis=1)
        ic(p_w_min)

        #find the index of p_w_min which is the hp
        hp_p_w_min = np.zeros(8)
        for i in range(1, 8):
            for j in range(len(self.hp)):
                if p_w[i, j] - p_w_min[i] < 0.001:
                    hp_p_w_min[i] = j * 0.01

        hp_p_w_min[0] = self.hp_threshold

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

    """

    
    n = 250
    w_s = np.linspace(100, 9000, n)
    constrains_name = ['stall speed', 'take off', 'cruise', 'service ceiling', 'level turn @3000m',
                       'climb @S-L', 'climb @3000m', 'climb @7000m', 'feasible region-hybrid', 'feasible region-conventional']
    color = ['k', 'c',  'b', 'g', 'y', 'plum', 'violet', 'm']
    methods = [ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun,
               ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun,
               ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_electric,
               ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_electric,
               ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun,
               ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun,
               ca_pd_12.ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun,
               ca_pd_12.ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun]
    m = constrains.shape[0]
    p_w = np.zeros([m, n, 8])


    # plots
    fig, ax = plt.subplots(3, 2, sharey=True, sharex=True, figsize=(10, 10))
    ax = ax.flatten()
    for k in range(8):
        for i in range(m):
            for j in range(n):
                h = constrains[i, 0]
                v = constrains[i, 1]
                beta = constrains[i, 2]
                hp = hp_p_w_min[i]

                # calculate p_w
                if k < 4:
                    problem = methods[k](h, v, beta, w_s[j], hp)
                    if i >= 5:
                        p_w[i, j,
                            k] = problem.allFuncs[-1](problem, roc=15 - 5 * (i - 5))
                    else:
                        p_w[i, j, k] = problem.allFuncs[i](problem)
                elif k > 5:
                    problem = methods[k](h, v, beta, w_s[j], Hp=0)
                    if i >= 5:
                        p_w[i, j,
                            k] = problem.allFuncs[-1](problem, roc=15 - 5 * (i - 5))
                    else:
                        p_w[i, j, k] = problem.allFuncs[i](problem)
                elif k == 4:
                    if i == 0:
                        problem = methods[k](h, v, beta, w_s[j], hp)
                        p_w[i, j, k] = problem.allFuncs[i](problem)
                    else:
                        p_w[i, j, k] = p_w[i, j, 0] + p_w[i, j, 2]
                else:
                    if i == 0:
                        problem = methods[k](h, v, beta, w_s[j], hp)
                        p_w[i, j, k] = problem.allFuncs[i](problem)
                    else:
                        p_w[i, j, k] = p_w[i, j, 1] + p_w[i, j, 3]

            if k <= 5:
                if i == 0:
                    ax[k].plot(p_w[i, :, k], np.linspace(0, 100, n),
                               linewidth=1, color=color[i], label=constrains_name[i])
                else:
                    ax[k].plot(w_s, p_w[i, :, k], color=color[i],
                               linewidth=1, alpha=1, label=constrains_name[i])
            else:
                if i == 0:
                    ax[k-2].plot(p_w[i, :, k], np.linspace(
                        0, 100, n), color=color[i], linewidth=1, alpha=0.5, linestyle='--')
                else:
                    ax[k-2].plot(w_s, p_w[i, :, k], color=color[i],
                                 linewidth=1, alpha=0.5, linestyle='--')

        if k <= 5:
            p_w[0, :, k] = 10 ** 10 * (w_s - p_w[0, 0, k])
            ax[k].fill_between(w_s, np.amax(p_w[0:m, :, k], axis=0),
                               150, color='b', alpha=0.5, label=constrains_name[-2])
            ax[k].set_xlim(200, 9000)
            ax[k].set_ylim(0, 100)
            ax[k].grid()
        else:
            p_w[0, :, k] = 10 ** 10 * (w_s - p_w[0, 0, k])
            ax[k-2].fill_between(w_s, np.amax(p_w[0:m, :, k], axis=0),
                                 150, color='r', alpha=0.5, label=constrains_name[-1])
            ax[k-2].plot(6012, 72, 'r*', markersize=5,
                         label='True Conventional')

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.125, 0.02, 0.75, 0.25), loc="lower left",
               mode="expand", borderaxespad=0, ncol=4, frameon=False)
    hp = constrains[:, 3]
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
    """
