# author: Bao Li # 
# Georgia Institute of Technology #
# import pyatmos as am
import numpy as np
import matplotlib.pylab as plt

"""
The unit use is IS standard
"""


class ConstrainsAnalysis:
    """This is a power-based master constraints analysis"""

    def __init__(self, tau, beta, alpha, K1, K2, C_D0, C_DR, wing_load):
        """

        :param tau: power fraction of i_th power path
        :param beta: power lapse ratio
        :param alpha: weight fraction
        :param K1: drag polar coefficient for 2nd order term
        :param K2: drag polar coefficient for 1st order term
        :param C_D0: the drag coefficient at zero lift
        :param C_DR: additional drag caused, for example, by external stores,
        braking parachutes or flaps, or temporary external hardware

        :return:
            power load: P_WTO
        """

        self.tau = tau
        self.beta = beta
        self.alpha = alpha

        self.K1 = K1
        self.K2 = K2
        self.C_D0 = C_D0
        self.C_DR = C_DR

        self.W_S = wing_load
        self.g0 = 9.80665

        self.coeff = self.tau * self.beta / self.alpha

    def master_equation(self, n, dh_dt, dV_dt, V, rho):
        q = 0.5 * rho * V ** 2

        linear_term = self.K1 * n ** 2 * self.beta / q
        inverse_term = (self.C_D0 + self.C_DR) * q / self.beta
        constant_term = self.K2 * n + dh_dt / V + dV_dt / self.g0
        # print(linear_term,'\n', inverse_term, '\n', constant_term)

        P_WTO = self.coeff * (linear_term * self.W_S + inverse_term / self.W_S + constant_term) * V
        return P_WTO

    def cruise(self, V=250, rho=0.4135):
        """
        A320NEO velocity for cruise 250 m/s at altitude 10000 m
        U.S. Standard Atmosphere Air Properties - SI Units at 10000 m is 1.225 kg/m^3
        """

        P_WTO = ConstrainsAnalysis.master_equation(self, n=1, dh_dt=0, dV_dt=0, V=V, rho=rho)
        return P_WTO

    def constant_speed_climb(self, dh_dt=10, V=150, rho=0.5):
        """
        A320neo climb speed 2000 ft/min, which is about 10 m/s
        assume clime at 300 knots, which is about 150 m/s
        assume air density equal to 0.5 kg/m^3
        """

        P_WTO = ConstrainsAnalysis.master_equation(self, n=1, dh_dt=dh_dt, dV_dt=0, V=V, rho=rho)
        return P_WTO

    def turn(self, V=150, turn_rate=3, rho=1):
        """
        assume 2 min for 360 degree turn, which is 3 degree/seconds
        assume turn at 300 knots, which is about 150 m/s
        assume air density is the density at 1000-2000 altitude which equal to 1 kg/m^3
        """

        load_factor = (1 + ((turn_rate * np.pi / 180) * V / self.g0) ** 2) ** 0.5
        P_WTO = ConstrainsAnalysis.master_equation(self, n=load_factor, dh_dt=0, dV_dt=0, V=V, rho=rho)
        return P_WTO

    def take_off_ground_roll(self, s_G=1500, K_TO=1.4, rho=1.225, CL_max=1.5):
        """
        A320neo take-off speed is about 150 knots, which is about 75 m/s
        required runway length is about 2000 m
        U.S. Standard Atmosphere Air Properties - SI Units at sea level is 1.225 kg/m^3
        K_TO is a constant greater than one set to 1.2 (generally specified by appropriate flying regulations)
        """

        P_WTO = 2/3*self.coeff / self.beta * (1 / s_G) * (1 / (rho*self.g0))*K_TO**2*self.W_S ** (3 / 2)

        P_WTO1 = self.coeff / self.beta * (1 / s_G) * (1 / (3 * self.g0)) \
                * (2 * K_TO ** 2 / (rho * CL_max)) ** (3 / 2) \
                * self.W_S ** (3 / 2)
        return P_WTO


if __name__ == "__main__":
    nn = 100
    wingload = np.linspace(1000, 6000, nn)
    P_W_cruise = np.zeros(nn)
    P_W_constant_speed_climb = np.zeros(nn)
    P_W_turn = np.zeros(nn)
    P_W_take_off_ground_roll = np.zeros(nn)

    for i in range(nn):
        problem = ConstrainsAnalysis(tau=0.5, beta=0.97, alpha=1, K1=0.2, K2=0,
                                     C_D0=0.02, C_DR=0.006, wing_load=wingload[i])
        P_W_cruise[i] = problem.cruise()
        P_W_constant_speed_climb[i] = problem.constant_speed_climb()
        P_W_turn[i] = problem.turn()
        P_W_take_off_ground_roll[i] = problem.take_off_ground_roll()

    plt.figure(figsize=(8, 6))
    plt.plot(wingload, P_W_cruise, 'b-', linewidth=1.5, label='cruise')
    plt.plot(wingload, P_W_constant_speed_climb, 'k-', linewidth=1.5, label='constant speed climb')
    plt.plot(wingload, P_W_turn, 'g-', linewidth=1.5, label='turn')
    plt.plot(wingload, P_W_take_off_ground_roll, 'r-', linewidth=1.5, label='take off ground roll')
    plt.xlabel('Wing Load: $W_{TO}$/S (N/${m^2}$)')
    plt.ylabel('Power-to-Load: $T_{SL}$/$W_{TO}$ (W/N)')
    plt.title('Constraint Analysis')
    plt.legend(loc=0)
    plt.grid()
    plt.show()
