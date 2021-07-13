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


"""
The unit use is IS standard
"""


class ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun:
    """This is a power-based master constraints analysis"""

    def __init__(self, altitude, velocity, beta, wing_load, Hp=0.5, number_of_motor=12, C_DR=0):
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
        self.hp = 1 - Hp
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

        self.coefficient = self.beta * self.v / self.alpha

        # Estimation of ΔCL and ΔCD
        pd = ad.aerodynamics_with_pd(
            self.h, self.v, Hp=self.hp, n=n, W_S=self.w_s)
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

    def cruise(self):
        p_w = ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun.master_equation(
            self, n=1, dh_dt=0, dV_dt=0)
        return p_w

    def climb(self, roc):
        p_w = ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun.master_equation(
            self, n=1, dh_dt=roc, dV_dt=0)
        return p_w

    def level_turn(self, turn_rate=3, v=100):
        """
        assume 2 min for 360 degree turn, which is 3 degree/seconds
        assume turn at 300 knots, which is about 150 m/s
        """
        load_factor = (1 + ((turn_rate * np.pi / 180)
                       * v / self.g0) ** 2) ** 0.5
        p_w = ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun.master_equation(
            self, n=load_factor, dh_dt=0, dV_dt=0)
        return p_w

    def take_off(self):
        """
        A320neo take-off speed is about 150 knots, which is about 75 m/s
        required runway length is about 2000 m
        K_TO is a constant greater than one set to 1.2 (generally specified by appropriate flying regulations)
        """
        Cl_max_to = 2.3  # 2.3
        K_TO = 1.2  # V_TO / V_stall
        s_G = 1266
        p_w = 2 / 3 * self.coefficient / self.v * self.beta * K_TO ** 2 / (
            s_G * self.rho * self.g0 * Cl_max_to) * self.w_s ** (
            3 / 2)
        return p_w

    def stall_speed(self, V_stall_to=65, Cl_max_to=2.3):
        V_stall_ld = 62

        Cl_max_ld = 2.87

        W_S_1 = 1 / 2 * self.rho * V_stall_to ** 2 * \
            (Cl_max_to + self.delta_cl)
        W_S_2 = 1 / 2 * self.rho * V_stall_ld ** 2 * \
            (Cl_max_ld + self.delta_cl)

        W_S = min(W_S_1, W_S_2)
        return W_S

    def service_ceiling(self, roc=0.5):
        p_w = ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun.master_equation(
            self, n=1, dh_dt=roc, dV_dt=0)
        return p_w

    allFuncs = [take_off, stall_speed, cruise,
                service_ceiling, level_turn, climb]


class ConstrainsAnalysis_Mattingly_Method_with_DP_electric:
    """This is a power-based master constraints analysis
    
    the difference between turbofun and electric for constrains analysis:
    1. assume the thrust_lapse = 1 for electric propution
    2. hp = 1 - hp_turbofun
    """

    def __init__(self, altitude, velocity, beta, wing_load, Hp=0.5, number_of_motor=12, C_DR=0):
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
        self.hp = Hp  # this is the difference part compare with turbofun
        self.n = number_of_motor

        # power lapse ratio
        self.alpha = 1  # this is the difference part compare with turbofun

        self.k1 = ad.aerodynamics_without_pd(self.h, self.v).K1()
        self.k2 = ad.aerodynamics_without_pd(self.h, self.v).K2()
        self.cd0 = ad.aerodynamics_without_pd(self.h, self.v).CD_0()
        self.cdr = C_DR

        self.w_s = wing_load
        self.g0 = 9.80665

        self.coefficient = self.beta * self.v / self.alpha

        # Estimation of ΔCL and ΔCD
        pd = ad.aerodynamics_with_pd(
            self.h, self.v, Hp=self.hp, n=n, W_S=self.w_s)
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

    def cruise(self):
        p_w = ConstrainsAnalysis_Mattingly_Method_with_DP_electric.master_equation(
            self, n=1, dh_dt=0, dV_dt=0)
        return p_w

    def climb(self, roc):
        p_w = ConstrainsAnalysis_Mattingly_Method_with_DP_electric.master_equation(
            self, n=1, dh_dt=roc, dV_dt=0)
        return p_w

    def level_turn(self, turn_rate=3, v=100):
        """
        assume 2 min for 360 degree turn, which is 3 degree/seconds
        assume turn at 300 knots, which is about 150 m/s
        """
        load_factor = (1 + ((turn_rate * np.pi / 180)
                       * v / self.g0) ** 2) ** 0.5
        p_w = ConstrainsAnalysis_Mattingly_Method_with_DP_electric.master_equation(
            self, n=load_factor, dh_dt=0, dV_dt=0)
        return p_w

    def take_off(self):
        """
        A320neo take-off speed is about 150 knots, which is about 75 m/s
        required runway length is about 2000 m
        K_TO is a constant greater than one set to 1.2 (generally specified by appropriate flying regulations)
        """
        Cl_max_to = 2.3  # 2.3
        K_TO = 1.2  # V_TO / V_stall
        s_G = 1266
        p_w = 2 / 3 * self.coefficient / self.v * self.beta * K_TO ** 2 / (
            s_G * self.rho * self.g0 * Cl_max_to) * self.w_s ** (
            3 / 2)
        return p_w

    def stall_speed(self, V_stall_to=65, Cl_max_to=2.3):
        V_stall_ld = 62

        Cl_max_ld = 2.87

        W_S_1 = 1 / 2 * self.rho * V_stall_to ** 2 * \
            (Cl_max_to + self.delta_cl)
        W_S_2 = 1 / 2 * self.rho * V_stall_ld ** 2 * \
            (Cl_max_ld + self.delta_cl)

        W_S = min(W_S_1, W_S_2)
        return W_S

    def service_ceiling(self, roc=0.5):
        p_w = ConstrainsAnalysis_Mattingly_Method_with_DP_electric.master_equation(
            self, n=1, dh_dt=roc, dV_dt=0)
        return p_w

    allFuncs = [take_off, stall_speed, cruise,
                service_ceiling, level_turn, climb]


class ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun:
    """This is a power-based master constraints analysis based on Gudmundsson_method"""

    def __init__(self, altitude, velocity, beta, wing_load, Hp=0.5, number_of_motor=12, e=0.75, AR=10.3):
        """

        :param beta: weight fraction
        :param e: wing planform efficiency factor is between 0.75 and 0.85, no more than 1
        :param AR: wing aspect ratio, normally between 7 and 10

        :return:
            power load: P_WTO
        """

        self.h = altitude
        self.v = velocity
        self.beta = beta
        self.w_s = wing_load
        self.g0 = 9.80665

        self.beta = beta
        self.hp = 1 - Hp
        self.n = number_of_motor

        self.rho = atm.atmosphere(geometric_altitude=self.h).density()

        # power lapse ratio
        self.alpha = thrust_lapse.thrust_lapse_calculation(altitude=self.h,
                                                           velocity=self.v).high_bypass_ratio_turbofan()
        h = 2.43  # height of winglets
        b = 35.8
        # equation 9-88, If the wing has winglets the aspect ratio should be corrected
        ar_corr = AR * (1 + 1.9 * h / b)
        self.k = 1 / (np.pi * ar_corr * e)
        self.coefficient = self.beta * self.v / self.alpha

        # Estimation of ΔCL and ΔCD
        pd = ad.aerodynamics_with_pd(
            self.h, self.v, Hp=self.hp, n=n, W_S=self.w_s)
        self.q = 0.5 * self.rho * self.v ** 2
        cl = self.beta * self.w_s / self.q
        self.delta_cl = pd.delta_lift_coefficient(cl)
        self.delta_cd0 = pd.delta_CD_0()

        # TABLE 3-1 Typical Aerodynamic Characteristics of Selected Classes of Aircraft
        cd_min = 0.02
        cd_to = 0.03
        cl_to = 0.8

        self.v_to = 68
        self.s_g = 1480
        self.mu = 0.04

        self.cd_min = cd_min + self.delta_cd0
        self.cl = cl + self.delta_cl

        self.cd_to = cd_to + self.delta_cd0
        self.cl_to = cl_to + self.delta_cl

    def cruise(self):
        p_w = self.q / self.w_s * (self.cd_min + self.k * self.cl ** 2)
        return p_w * self.coefficient

    def climb(self, roc):
        p_w = roc / self.v + self.q * self.cd_min / self.w_s + self.k * self.cl
        return p_w * self.coefficient

    def level_turn(self, turn_rate=3, v=100):
        """
        assume 2 min for 360 degree turn, which is 3 degree/seconds
        assume turn at 100 m/s
        """
        load_factor = (1 + ((turn_rate * np.pi / 180)
                       * v / self.g0) ** 2) ** 0.5
        q = 0.5 * self.rho * v ** 2
        p_w = q / self.w_s * (self.cd_min + self.k *
                              (load_factor / q * self.w_s + self.delta_cl) ** 2)
        return p_w * self.coefficient

    def take_off(self):
        q = self.q / 2
        p_w = self.v_to ** 2 / (2 * self.g0 * self.s_g) + q * self.cd_to / self.w_s + self.mu * (
            1 - q * self.cl_to / self.w_s)
        return p_w * self.coefficient

    def service_ceiling(self, roc=0.5):
        vy = (2 / self.rho * self.w_s *
              (self.k / (3 * self.cd_min)) ** 0.5) ** 0.5
        q = 0.5 * self.rho * vy ** 2
        p_w = roc / vy + q / self.w_s * \
            (self.cd_min + self.k * (self.w_s / q + self.delta_cl) ** 2)
        # p_w = roc / (2 / self.rho * self.w_s * (self.k / (3 * self.cd_min)) ** 0.5) ** 0.5 + 4 * (
        #         self.k * self.cd_min / 3) ** 0.5
        return p_w * self.coefficient

    def stall_speed(self, V_stall_to=65, Cl_max_to=2.3):
        V_stall_ld = 62

        Cl_max_ld = 2.87

        W_S_1 = 1 / 2 * self.rho * V_stall_to ** 2 * \
            (Cl_max_to + self.delta_cl)
        W_S_2 = 1 / 2 * self.rho * V_stall_ld ** 2 * \
            (Cl_max_ld + self.delta_cl)

        W_S = min(W_S_1, W_S_2)
        return W_S

    allFuncs = [take_off, stall_speed, cruise,
                service_ceiling, level_turn, climb]


class ConstrainsAnalysis_Gudmundsson_Method_with_DP_electric:
    """This is a power-based master constraints analysis based on Gudmundsson_method
    
    the difference between turbofun and electric for constrains analysis:
    1. assume the thrust_lapse = 1 for electric propution
    2. hp = 1 - hp_turbofun
    """

    def __init__(self, altitude, velocity, beta, wing_load, Hp=0.5, number_of_motor=12, e=0.75, AR=10.3):
        """

        :param beta: weight fraction
        :param e: wing planform efficiency factor is between 0.75 and 0.85, no more than 1
        :param AR: wing aspect ratio, normally between 7 and 10

        :return:
            power load: P_WTO
        """

        self.h = altitude
        self.v = velocity
        self.beta = beta
        self.w_s = wing_load
        self.g0 = 9.80665

        self.beta = beta
        self.hp = Hp  # this is the difference part compare with turbofun
        self.n = number_of_motor

        self.rho = atm.atmosphere(geometric_altitude=self.h).density()

        # power lapse ratio
        self.alpha = 1  # this is the difference part compare with turbofun

        h = 2.43  # height of winglets
        b = 35.8
        # equation 9-88, If the wing has winglets the aspect ratio should be corrected
        ar_corr = AR * (1 + 1.9 * h / b)
        self.k = 1 / (np.pi * ar_corr * e)
        self.coefficient = self.beta * self.v / self.alpha

        # Estimation of ΔCL and ΔCD
        pd = ad.aerodynamics_with_pd(
            self.h, self.v, Hp=self.hp, n=n, W_S=self.w_s)
        self.q = 0.5 * self.rho * self.v ** 2
        cl = self.beta * self.w_s / self.q
        self.delta_cl = pd.delta_lift_coefficient(cl)
        self.delta_cd0 = pd.delta_CD_0()

        # TABLE 3-1 Typical Aerodynamic Characteristics of Selected Classes of Aircraft
        cd_min = 0.02
        cd_to = 0.03
        cl_to = 0.8

        self.v_to = 68
        self.s_g = 1480
        self.mu = 0.04

        self.cd_min = cd_min + self.delta_cd0
        self.cl = cl + self.delta_cl

        self.cd_to = cd_to + self.delta_cd0
        self.cl_to = cl_to + self.delta_cl

    def cruise(self):
        p_w = self.q / self.w_s * (self.cd_min + self.k * self.cl ** 2)
        return p_w * self.coefficient

    def climb(self, roc):
        p_w = roc / self.v + self.q * self.cd_min / self.w_s + self.k * self.cl
        return p_w * self.coefficient

    def level_turn(self, turn_rate=3, v=100):
        """
        assume 2 min for 360 degree turn, which is 3 degree/seconds
        assume turn at 100 m/s
        """
        load_factor = (1 + ((turn_rate * np.pi / 180)
                       * v / self.g0) ** 2) ** 0.5
        q = 0.5 * self.rho * v ** 2
        p_w = q / self.w_s * (self.cd_min + self.k *
                              (load_factor / q * self.w_s + self.delta_cl) ** 2)
        return p_w * self.coefficient

    def take_off(self):
        q = self.q / 2
        p_w = self.v_to ** 2 / (2 * self.g0 * self.s_g) + q * self.cd_to / self.w_s + self.mu * (
            1 - q * self.cl_to / self.w_s)
        return p_w * self.coefficient

    def service_ceiling(self, roc=0.5):
        vy = (2 / self.rho * self.w_s *
              (self.k / (3 * self.cd_min)) ** 0.5) ** 0.5
        q = 0.5 * self.rho * vy ** 2
        p_w = roc / vy + q / self.w_s * \
            (self.cd_min + self.k * (self.w_s / q + self.delta_cl) ** 2)
        # p_w = roc / (2 / self.rho * self.w_s * (self.k / (3 * self.cd_min)) ** 0.5) ** 0.5 + 4 * (
        #         self.k * self.cd_min / 3) ** 0.5
        return p_w * self.coefficient

    def stall_speed(self, V_stall_to=65, Cl_max_to=2.3):
        V_stall_ld = 62

        Cl_max_ld = 2.87

        W_S_1 = 1 / 2 * self.rho * V_stall_to ** 2 * \
            (Cl_max_to + self.delta_cl)
        W_S_2 = 1 / 2 * self.rho * V_stall_ld ** 2 * \
            (Cl_max_ld + self.delta_cl)

        W_S = min(W_S_1, W_S_2)
        return W_S

    allFuncs = [take_off, stall_speed, cruise,
                service_ceiling, level_turn, climb]

if __name__ == "__main__":
    n = 100
    w_s = np.linspace(100, 9000, n)
    constrains_name = ['take off', 'stall speed', 'cruise', 'service ceiling', 'level turn @3000m',
                       'climb @S-L', 'climb @3000m', 'climb @7000m']
    constrains = np.array([[0, 68, 0.988, 0.5], [0, 80, 1, 0.5], [11300, 230, 0.948, 0.8],
                           [11900, 230, 0.78, 0.5], [3000, 100, 0.984, 0.8], [0, 100, 0.984, 0.5],
                           [3000, 200, 0.975, 0.6], [7000, 230, 0.96, 0.8]])
    color = ['c', 'k', 'b', 'g', 'y', 'plum', 'violet', 'm']
    label = ['feasible region with PD', 'feasible region with PD', 'feasible region Gudmundsson',
             'feasible region without PD', 'feasible region without PD', 'feasible region Mattingly']
    methods = [ConstrainsAnalysis_Mattingly_Method_with_DP_turbofun,
               ConstrainsAnalysis_Mattingly_Method_with_DP_electric,
               ca.ConstrainsAnalysis_Mattingly_Method,
               ca.ConstrainsAnalysis_Mattingly_Method,
               ConstrainsAnalysis_Gudmundsson_Method_with_DP_turbofun, 
               ConstrainsAnalysis_Gudmundsson_Method_with_DP_electric,
               ca.ConstrainsAnalysis_Gudmundsson_Method,
               ca.ConstrainsAnalysis_Gudmundsson_Method]
    subtitle = ["1", '2']
    m = constrains.shape[0]
    p_w = np.zeros([m, n, 8])

    # plots
    fig, ax = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(10, 10))
    ax = ax.flatten()
    for k in range(8):
        if k == 2:
            p_w[:, :, k] = p_w[:, :, 0] + p_w[:, :, 1]
        elif k == 6:
            p_w[:, :, k] = p_w[:, :, 4] + p_w[:, :, 5]

        for i in range(m):
            for j in range(n):
                h = constrains[i, 0]
                v = constrains[i, 1]
                beta = constrains[i, 2]
                hp = constrains[i, 3]

                if k == 0 or k == 1 or k == 4 or k == 5:
                    problem = methods[k](h, v, beta, w_s[j], hp)
                elif k == 3 or k == 7:
                    problem = methods[k](h, v, beta, w_s[j])

                if i >= 5:
                    p_w[i, j, k] = problem.allFuncs[-1](problem, roc=15 - 5 * (i - 5))
                else:
                    p_w[i, j, k] = problem.allFuncs[i](problem)
                
            if i == 1:
                ax[k].plot(p_w[i, :, k], np.linspace(0, 250, n), color=color[i], label=constrains_name[i])
            else:
                ax[k].plot(w_s, p_w[i, :, k], color=color[i], label=constrains_name[i])
        p_w[1, :, k] = 200 / (p_w[1, -1, k] - p_w[1, 20, k]) * (w_s - p_w[1, 2, k])
        ax[k].fill_between(w_s, np.amax(p_w[0:m - 1, :, k], axis=0), 200, color='b', alpha=0.25)

        # plt.xlabel('Wing Load: $W_{TO}$/S (N/${m^2}$)')
        # plt.ylabel('Power-to-Load: $P_{SL}$/$W_{TO}$ (W/N)')
        ax[k].legend(bbox_to_anchor=(1.002, 1), loc="upper left")
        # plt.gca().add_artist(l1)
        # axs[k].xlim(100, 9000)
        # axs[k].ylim(0, 200)
        # plt.tight_layout()
        ax[k].grid()
    plt.show()
