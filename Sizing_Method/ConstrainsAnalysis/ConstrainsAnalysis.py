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


class ConstrainsAnalysis:
    """This is a power-based master constraints analysis"""

    def __init__(self, altitude, velocity, tau, beta, wing_load, C_DR=0):
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
        """
        A320NEO velocity for cruise 250 m/s at altitude 10000 m
        U.S. Standard Atmosphere Air Properties - SI Units at 10000 m is 1.225 kg/m^3
        """

        P_WTO = ConstrainsAnalysis.master_equation(self, n=1, dh_dt=0, dV_dt=0)
        return P_WTO

    def constant_speed_climb(self, h_initial, h_final, delta_t):
        """
        A320neo climb speed 2000 ft/min, which is about 10 m/s
        assume clime at 300 knots, which is about 150 m/s
        """
        dh_dt = (h_final - h_initial) / delta_t

        if dh_dt <= 0:
            print("it is not climb")

        P_WTO = ConstrainsAnalysis.master_equation(self, n=1, dh_dt=dh_dt, dV_dt=0)
        return P_WTO

    def acceleration_climb(self, v_initial, v_final, h_initial, h_final, delta_t):
        """
        A320neo climb speed 2000 ft/min, which is about 10 m/s
        assume clime at 300 knots, which is about 150 m/s
        """
        dh_dt = (h_final - h_initial) / delta_t
        dv_dt = (v_final - v_initial) / delta_t

        if dv_dt <= 0:
            print("it is no acceleration")

        if dh_dt <= 0:
            print("it is not climb")

        P_WTO = ConstrainsAnalysis.master_equation(self, n=1, dh_dt=dh_dt, dV_dt=dv_dt)
        return P_WTO

    def turn(self, turn_rate=3):
        """
        assume 2 min for 360 degree turn, which is 3 degree/seconds
        assume turn at 300 knots, which is about 150 m/s
        """

        load_factor = (1 + ((turn_rate * np.pi / 180) * self.v / self.g0) ** 2) ** 0.5
        P_WTO = ConstrainsAnalysis.master_equation(self, n=load_factor, dh_dt=0, dV_dt=0)
        return P_WTO

    def horizontal_acceleration(self, v_initial, v_final, delta_t):
        dv_dt = (v_final - v_initial) / delta_t

        if dv_dt <= 0:
            print("it is no acceleration")

        P_WTO = ConstrainsAnalysis.master_equation(self, n=1, dh_dt=0, dV_dt=dv_dt)
        return P_WTO

    def take_off(self):
        """
        A320neo take-off speed is about 150 knots, which is about 75 m/s
        required runway length is about 2000 m
        K_TO is a constant greater than one set to 1.2 (generally specified by appropriate flying regulations)
        """
        Cl_max_to = 2.3
        K_TO = 1.2  # V_TO / V_stall
        s_G = 1266
        P_WTO = 2 / 3 * self.coeff * self.beta * K_TO ** 2 / (s_G * self.rho * self.g0 * Cl_max_to) * self.W_S ** (
                    3 / 2)
        return P_WTO

    def landing(self):
        Cl_max_ld = 2.87
        K_TD = 1.15  # V_TD / V_stall
        s_B = 687
        P_WTO = 2 / 3 * self.coeff * self.beta * K_TD ** 2 / (s_B * self.rho * self.g0 * Cl_max_ld) * self.W_S ** (
                    3 / 2)
        return P_WTO

    def stall(self):
        V_stall_to = 65
        V_stall_ld = 62

        Cl_max_to = 2.3
        Cl_max_ld = 2.87

        W_S_1 = 1 / 2 * self.rho * V_stall_to ** 2 * Cl_max_to
        W_S_2 = 1 / 2 * self.rho * V_stall_ld ** 2 * Cl_max_ld

        W_S = min(W_S_1, W_S_2)
        return W_S


if __name__ == "__main__":
    nn = 100
    wingload = np.linspace(1000, 7000, nn)

    take_off = np.zeros(nn)
    landing = np.zeros(nn)
    stall_speed = np.zeros(nn)

    e = np.zeros(nn)
    j = np.zeros(nn)
    l = np.zeros(nn)


    b = np.zeros(nn)
    P_W_cruise = np.zeros(nn)
    P_W_constant_speed_climb = np.zeros(nn)
    P_W_turn = np.zeros(nn)
    P_W_take_off_ground_roll = np.zeros(nn)
    W_S_stall = np.zeros(nn)

    for i in range(nn):
        take_off[i] = ConstrainsAnalysis(altitude=0, velocity=78 / 2, tau=1, beta=0.988,
                                         wing_load=wingload[i]).take_off()
        landing[i] = ConstrainsAnalysis(altitude=0, velocity=71 / 2, tau=1, beta=0.7, wing_load=wingload[i]).landing()
        stall_speed[i] = ConstrainsAnalysis(altitude=0, velocity=80, tau=1, beta=0.948, wing_load=wingload[i]).stall()
        e[i] = ConstrainsAnalysis(altitude=5, velocity=80, tau=1, beta=0.985, wing_load=wingload[i]).acceleration_climb(78, 85, 0, 10, 7)
        j[i] = ConstrainsAnalysis(altitude=(10000-3000)/2, velocity=(225-200)/2, tau=1, beta=0.985, wing_load=wingload[i]).acceleration_climb(200, 225, 3000, 10000, 1200)
        l[i] = ConstrainsAnalysis(altitude=10000, velocity=235, tau=1, beta=0.948, wing_load=wingload[i]).cruise()

        # b = ConstrainsAnalysis(altitude=0, velocity=78/2, tau=1, beta=0.97, wing_load=wingload[i]).horizontal_acceleration(0, 78, 30)

    plt.figure(figsize=(8, 6))
    plt.plot(wingload, take_off, 'b-', linewidth=1.5, label='take off')
    plt.plot(wingload, landing, 'k-', linewidth=1.5, label='landing')
    plt.plot(wingload, j, 'g-', linewidth=1.5, label='turn')
    plt.plot(wingload, l, 'g-', linewidth=1.5, label='turn')
    # plt.plot(wingload, b, 'g-', linewidth=1.5, label='turn')
    plt.plot(wingload, e, 'y-', linewidth=1.5, label='take off ground roll')
    plt.plot(stall_speed, np.linspace(0, 250, nn), 'r-', linewidth=1.5, label='landing')
    plt.xlabel('Wing Load: $W_{TO}$/S (N/${m^2}$)')
    plt.ylabel('Power-to-Load: $P_{SL}$/$W_{TO}$ (W/N)')
    plt.title('Constraint Analysis')
    plt.legend(loc=0)
    plt.grid()
    plt.show()
