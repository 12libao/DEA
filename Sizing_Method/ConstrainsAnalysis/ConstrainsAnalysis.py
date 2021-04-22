# author: Bao Li #
# Georgia Institute of Technology #

import numpy as np
import matplotlib.pylab as plt
import Sizing_Method.Other.US_Standard_Atmosphere_1976 as atm
import Sizing_Method.Aerodynamics.ThrustLapse as thrust_lapse
import Sizing_Method.Aerodynamics.Aerodynamics as ad

"""
The unit use is IS standard
"""


class ConstrainsAnalysis_Mattingly_Method:
    """This is a power-based master constraints analysis"""

    def __init__(self, altitude, velocity, beta, wing_load, tau=1, C_DR=0):
        """

        :param tau: power fraction of i_th power path
        :param beta: weight fraction
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

        self.tau = tau
        self.beta = beta

        # power lapse ratio
        self.alpha = thrust_lapse.thrust_lapse_calculation(altitude=self.h,
                                                           velocity=self.v).high_bypass_ratio_turbofan()

        self.K1 = ad.aerodynamics_without_pd(self.h, self.v).K1()
        self.K2 = ad.aerodynamics_without_pd(self.h, self.v).K2()
        self.C_D0 = ad.aerodynamics_without_pd(self.h, self.v).CD_0()
        self.C_DR = C_DR

        self.W_S = wing_load
        self.g0 = 9.80665

        self.coeff = self.tau * self.beta / self.alpha

    def master_equation(self, n, dh_dt, dV_dt):
        q = 0.5 * self.rho * self.v ** 2

        linear_term = self.K1 * n ** 2 * self.beta / q
        inverse_term = (self.C_D0 + self.C_DR) * q / self.beta
        constant_term = self.K2 * n + dh_dt / self.v + dV_dt / self.g0
        # print(linear_term,'\n', inverse_term, '\n', constant_term)

        P_WTO = self.coeff * (linear_term * self.W_S + inverse_term / self.W_S + constant_term) * self.v
        return P_WTO

    def cruise(self):
        P_WTO = ConstrainsAnalysis_Mattingly_Method.master_equation(self, n=1, dh_dt=0, dV_dt=0)
        return P_WTO

    def climb(self, roc):
        P_WTO = ConstrainsAnalysis_Mattingly_Method.master_equation(self, n=1, dh_dt=roc, dV_dt=0)
        return P_WTO

    def level_turn(self, turn_rate=3, v=100):
        """
        assume 2 min for 360 degree turn, which is 3 degree/seconds
        assume turn at 300 knots, which is about 150 m/s
        """
        load_factor = (1 + ((turn_rate * np.pi / 180) * v / self.g0) ** 2) ** 0.5
        P_WTO = ConstrainsAnalysis_Mattingly_Method.master_equation(self, n=load_factor, dh_dt=0, dV_dt=0)
        return P_WTO

    def horizontal_acceleration(self, v_initial, v_final, delta_t):
        dv_dt = (v_final - v_initial) / delta_t

        if dv_dt <= 0:
            print("it is no acceleration")

        P_WTO = ConstrainsAnalysis_Mattingly_Method.master_equation(self, n=1, dh_dt=0, dV_dt=dv_dt)
        return P_WTO

    def take_off(self):
        """
        A320neo take-off speed is about 150 knots, which is about 75 m/s
        required runway length is about 2000 m
        K_TO is a constant greater than one set to 1.2 (generally specified by appropriate flying regulations)
        """
        Cl_max_to = 2.3  # 2.3
        K_TO = 1.2  # V_TO / V_stall
        s_G = 1266
        P_WTO = 2 / 3 * self.coeff * self.beta * K_TO ** 2 / (s_G * self.rho * self.g0 * Cl_max_to) * self.W_S ** (
                3 / 2)
        return P_WTO

    def stall_speed(self):
        V_stall_to = 65
        V_stall_ld = 62

        Cl_max_to = 2.3
        Cl_max_ld = 2.87

        W_S_1 = 1 / 2 * self.rho * V_stall_to ** 2 * Cl_max_to
        W_S_2 = 1 / 2 * self.rho * V_stall_ld ** 2 * Cl_max_ld

        W_S = min(W_S_1, W_S_2)
        return W_S

    def service_ceiling(self, roc=0.5):
        P_WTO = ConstrainsAnalysis_Mattingly_Method.master_equation(self, n=1, dh_dt=roc, dV_dt=0)
        return P_WTO

    allFuncs = [take_off, stall_speed, cruise, service_ceiling, level_turn, climb]


class ConstrainsAnalysis_Gudmundsson_Method:
    """This is a power-based master constraints analysis based on Gudmundsson_method"""

    def __init__(self, altitude, velocity, beta, wing_load, tau=1, e=0.75, AR=10.3):
        """

        :param tau: power fraction of i_th power path
        :param beta: weight fraction
        :param e: wing planform efficiency factor is between 0.75 and 0.85, no more than 1
        :param AR: wing aspect ratio, normally between 7 and 10

        :return:
            power load: P_WTO
        """

        self.h = altitude
        self.v = velocity
        self.tau = tau
        self.beta = beta
        self.w_s = wing_load
        self.g0 = 9.80665

        self.rho = atm.atmosphere(geometric_altitude=self.h).density()
        self.q = 0.5 * self.rho * self.v ** 2

        # power lapse ratio
        self.alpha = thrust_lapse.thrust_lapse_calculation(altitude=self.h,
                                                           velocity=self.v).high_bypass_ratio_turbofan()
        h = 2.43  # height of winglets
        b = 35.8
        ar_corr = AR * (1 + 1.9 * h / b)  # equation 9-88, If the wing has winglets the aspect ratio should be corrected
        self.k = 1 / (np.pi * ar_corr * e)
        self.coefficient = self.tau * self.beta * self.v / self.alpha

        # TABLE 3-1 Typical Aerodynamic Characteristics of Selected Classes of Aircraft
        self.cd_min = 0.02
        self.cd_to = 0.03
        self.cl_to = 0.8

        self.v_to = 68
        self.s_g = 1480
        self.mu = 0.04

    def cruise(self):
        p_w = self.q * self.cd_min / self.w_s + self.k / self.q * self.w_s
        return p_w * self.coefficient

    def climb(self, roc):
        p_w = roc / self.v + self.q * self.cd_min / self.w_s + self.k / self.q * self.w_s
        return p_w * self.coefficient

    def level_turn(self, turn_rate=3, v=100):
        """
        assume 2 min for 360 degree turn, which is 3 degree/seconds
        assume turn at 100 m/s
        """
        load_factor = (1 + ((turn_rate * np.pi / 180) * v / self.g0) ** 2) ** 0.5
        q = 0.5 * self.rho * v ** 2
        p_w = q * (self.cd_min / self.w_s + self.k * (load_factor / q) ** 2 * self.w_s)
        return p_w * self.coefficient

    def take_off(self):
        q = self.q / 2
        p_w = self.v_to ** 2 / (2 * self.g0 * self.s_g) + q * self.cd_to / self.w_s + self.mu * (
                1 - q * self.cl_to / self.w_s)
        return p_w * self.coefficient

    def service_ceiling(self, roc=0.5):
        p_w = roc / (2 / self.rho * self.w_s * (self.k / (3 * self.cd_min)) ** 0.5) ** 0.5 + 4 * (
                self.k * self.cd_min / 3) ** 0.5
        return p_w * self.coefficient

    def stall_speed(self, V_stall_to=65, Cl_max_to=2.3):
        V_stall_ld = 62

        Cl_max_ld = 2.87

        W_S_1 = 1 / 2 * self.rho * V_stall_to ** 2 * Cl_max_to
        W_S_2 = 1 / 2 * self.rho * V_stall_ld ** 2 * Cl_max_ld

        W_S = min(W_S_1, W_S_2)
        return W_S

    allFuncs = [take_off, stall_speed, cruise, service_ceiling, level_turn, climb]


if __name__ == "__main__":

    """


    nn = 100
    wingload = np.linspace(100, 8000, nn)

    take_off = np.zeros(nn)
    landing = np.zeros(nn)
    stall_speed = np.zeros(nn)
    cruise = np.zeros(nn)
    divert = np.zeros(nn)
    loiter = np.zeros(nn)
    service_ceiling = np.zeros(nn)

    e = np.zeros(nn)
    f = np.zeros(nn)
    g = np.zeros(nn)
    h = np.zeros(nn)
    ii = np.zeros(nn)
    j = np.zeros(nn)
    takeoff2 = np.zeros(nn)
    takeoff1 = np.zeros(nn)
    k = np.zeros(nn)
    n = np.zeros(nn)

    cruise1 = np.zeros(nn)

    for i in range(nn):

        take_off[i] = ConstrainsAnalysis(altitude=0, velocity=73 / 2, tau=1, beta=0.988,
                                         wing_load=wingload[i]).take_off()
        landing[i] = ConstrainsAnalysis(altitude=0, velocity=63 / 2, tau=1, beta=0.7,
                                        wing_load=wingload[i]).landing()
        stall_speed[i] = ConstrainsAnalysis(altitude=0, velocity=80, tau=1, beta=1,
                                            wing_load=wingload[i]).stall()
        cruise[i] = ConstrainsAnalysis(altitude=10000, velocity=235, tau=1, beta=0.948,
                                       wing_load=wingload[i]).cruise()
        divert[i] = ConstrainsAnalysis(altitude=3000, velocity=165, tau=1, beta=0.77,
                                       wing_load=wingload[i]).cruise()
        loiter[i] = ConstrainsAnalysis(altitude=500, velocity=165, tau=1, beta=0.72,
                                       wing_load=wingload[i]).level_constant_velocity_turn()
        # service_ceiling[i] = ConstrainsAnalysis(altitude=10000, velocity=230, tau=1, beta=0.95,
        #                                        wing_load=wingload[i]).acceleration_climb(230, 230, 0, 1.0, 2)

        e[i] = ConstrainsAnalysis(altitude=5, velocity=78, tau=1, beta=0.985,
                                  wing_load=wingload[i]).acceleration_climb(73, 78, 0, 10, 3)
        f[i] = ConstrainsAnalysis(altitude=250, velocity=100, tau=1, beta=0.984,
                                  wing_load=wingload[i]).acceleration_climb(78, 100, 10, 500, 38)
        g[i] = ConstrainsAnalysis(altitude=500, velocity=125, tau=1, beta=0.982,
                                  wing_load=wingload[i]).horizontal_acceleration(100, 150, 100)
        h[i] = ConstrainsAnalysis(altitude=1500, velocity=157.5, tau=1, beta=0.981,
                                  wing_load=wingload[i]).acceleration_climb(150, 165, 500, 3000, 227)
        ii[i] = ConstrainsAnalysis(altitude=3000, velocity=182.5, tau=1, beta=0.975,
                                  wing_load=wingload[i]).horizontal_acceleration(165, 200, 70)
        j[i] = ConstrainsAnalysis(altitude=3000, velocity=200, tau=1, beta=0.973,
                                  wing_load=wingload[i]).acceleration_climb(200, 225, 3000, 10000, 1300)
        n[i] = ConstrainsAnalysis(altitude=1500, velocity=118, tau=1, beta=0.78,
                                  wing_load=wingload[i]).acceleration_climb(70, 165, 100, 3000, 270)


    plt.figure(figsize=(8, 6))
    plt.plot(wingload, take_off, label='takeoff')
    plt.plot(wingload, landing, label='landing')
    plt.plot(stall_speed, np.linspace(0, 250, nn), label='stall')
    plt.plot(wingload, cruise, label='cruise')
    plt.plot(wingload, divert, label='divert')
    plt.plot(wingload, loiter, label='loiter')
    plt.plot(wingload, service_ceiling, label='service_ceiling')
    plt.plot(wingload, e, label='e-AClimb')
    plt.plot(wingload, f, label='f-AClimb')
    plt.plot(wingload, g, label='g-A')
    plt.plot(wingload, h, label='h-AClimb')
    plt.plot(wingload, ii, label='i-A')
    plt.plot(wingload, j, label='j-AClimb')
    plt.plot(wingload, n, label='n-AClimb')
    plt.plot(wingload, takeoff1, label='takeoff1')
    plt.plot(wingload, takeoff2, label='takeoff2')
    plt.plot(wingload, cruise1, label='cruise1')

    plt.xlabel('Wing Load: $W_{TO}$/S (N/${m^2}$)')
    plt.ylabel('Power-to-Load: $P_{SL}$/$W_{TO}$ (W/N)')
    plt.title('Constraint Analysis')
    plt.legend(bbox_to_anchor=(1.002, 1), loc="upper left")
    plt.xlim(100, 8000)
    plt.ylim(0, 200)
    plt.tight_layout()
    plt.grid()
    plt.show()
    """

    n = 100
    w_s = np.linspace(100, 9000, n)
    constrains_name = ['take off', 'stall speed', 'cruise', 'service ceiling', 'level turn @3000m',
                       'climb @S-L', 'climb @3000m', 'climb @7000m']
    constrains = np.array([[0, 68, 0.988], [0, 80, 1], [11300, 230, 0.948],
                           [11900, 230, 0.948], [3000, 100, 0.984], [0, 100, 0.984],
                           [3000, 200, 0.975], [7000, 230, 0.96]])
    color = ['c', 'k', 'b', 'g', 'y', 'plum', 'violet', 'm']
    m = constrains.shape[0]
    p_w = np.zeros([2 * m, n])

    plt.figure(figsize=(10, 8))
    for i in range(m):
        for j in range(n):
            h = constrains[i, 0]
            v = constrains[i, 1]
            beta = constrains[i, 2]
            problem1 = ConstrainsAnalysis_Gudmundsson_Method(h, v, beta, w_s[j])
            problem2 = ConstrainsAnalysis_Mattingly_Method(h, v, beta, w_s[j])

            if i >= 5:
                p_w[i, j] = problem1.allFuncs[-1](problem1, roc=15 - 5 * (i - 5))
                p_w[i + m, j] = problem2.allFuncs[-1](problem2, roc=15 - 5 * (i - 5))
            else:
                p_w[i, j] = problem1.allFuncs[i](problem1)
                p_w[i + m, j] = problem2.allFuncs[i](problem2)

        if i == 1:
            pa, = plt.plot(p_w[i, :], np.linspace(0, 250, n), color=color[i], label=constrains_name[i])
            pb, = plt.plot(p_w[i + m, :], np.linspace(0, 250, n), color=color[i], linestyle='--')
            l1 = plt.legend([pa, pb], ['Gudmundsson method', 'Mattingly method'], loc="upper right")
        else:
            plt.plot(w_s, p_w[i, :], color=color[i], label=constrains_name[i])
            plt.plot(w_s, p_w[i + m, :], color=color[i], linestyle='--')

    plt.xlabel('Wing Load: $W_{TO}$/S (N/${m^2}$)')
    plt.ylabel('Power-to-Load: $P_{SL}$/$W_{TO}$ (W/N)')
    plt.title('Constraint Analysis-Normalized to Sea Level')
    plt.legend(bbox_to_anchor=(1.002, 1), loc="upper left")
    plt.gca().add_artist(l1)
    plt.xlim(100, 9000)
    plt.ylim(0, 200)
    plt.tight_layout()
    plt.grid()
    plt.show()