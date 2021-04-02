# author: Bao Li # 
# Georgia Institute of Technology #
"""Reference:

1: U.S. Standard Atmosphere, 1976, U.S. Government Printing Office, Washington, D.C., 1976.
2: Mattingly, Jack D., William H. Heiser, and David T. Pratt. Aircraft engine design. American Institute of Aeronautics and Astronautics, 2002.
3: https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere
4: https://en.wikipedia.org/wiki/Barometric_formula
"""

import numpy as np
import matplotlib.pylab as plt


class atmosphere:
    """this is the U.S. Standard Atmosphere 1976 based on the Barometric formula

    1: All of the following is limited to geometric altitudes below 86 km
    2: SI Units
    """

    def __init__(self, geometric_altitude):
        """
        :input h (m): geometric altitude

        :output P (Pa = N/m^2): pressure
                T (K): temperature
                rho (kg/m^3): density
                a (m/s): sound speed

                delta: dimensionless static pressure = P/P0
                theta: dimensionless static temperature = T/T0
                sigma: dimensionless static density = rho/rho0
        """

        # input:
        self.h = geometric_altitude

        # constant:
        self.r0 = 6356577.0  # the earth's radius (m)
        self.R_star = 8.3144598  # universal gas constant (J/mol.K)
        self.g0 = 9.80665  # gravitational acceleration (m/s^2)
        self.M = 0.0289644  # molar mass of Earth's air (kg/mol)

        # sea level (standard)
        self.T0 = 288.15  # sea level temperature (K)
        self.P0 = 101325  # sea level pressure (N/m^2)
        self.rho0 = 1.225  # sea level density (kg/m^3)
        self.a0 = 340.3  # sea level sound speed (m/s)

        # temperature lapse rate (K/m)
        self.lapse_rate = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])
        # height above sea level for each lapse rate (m)
        self.zi = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84853.0])
        # number of piecewise linear curve-fit
        self.n = len(self.lapse_rate)

    def geo_potential_altitude(self):
        z = self.r0 * self.h / (self.r0 + self.h)

        # decide the number of piecewise linear curve-fit
        i = 0
        while self.zi[i] < z:
            i = i + 1

        if i != 0:
            i = i - 1
        return z, i

    def ti(self):
        Ti = np.zeros(self.n)
        Ti[0] = self.T0

        for i in range(self.n - 1):
            Ti[i + 1] = Ti[i] + self.lapse_rate[i] * (self.zi[i + 1] - self.zi[i])
        return Ti

    def temperature(self):
        z, i = atmosphere.geo_potential_altitude(self)
        Ti = atmosphere.ti(self)
        T = Ti[i] + self.lapse_rate[i] * (z - self.zi[i])
        return T

    def pressure(self):
        Ti = atmosphere.ti(self)
        Pi = np.zeros(self.n)
        Pi[0] = self.P0

        for i in range(self.n - 1):
            if self.lapse_rate[i] != 0:
                Pi[i + 1] = Pi[i] * (Ti[i] / Ti[i + 1]) ** (self.g0 * self.M / (self.R_star * self.lapse_rate[i]))
            else:
                Pi[i + 1] = Pi[i] * np.exp(-self.g0 * self.M * (self.zi[i + 1] - self.zi[i]) / (self.R_star * Ti[i]))

        z, i = atmosphere.geo_potential_altitude(self)
        T = atmosphere.temperature(self)

        if self.lapse_rate[i] != 0:
            P = Pi[i] * (Ti[i] / T) ** (self.g0 * self.M / (self.R_star * self.lapse_rate[i]))
        else:
            P = Pi[i] * np.exp(-self.g0 * self.M * (z - self.zi[i]) / (self.R_star * Ti[i]))
        return P

    def density(self):
        P = atmosphere.pressure(self)
        T = atmosphere.temperature(self)
        rho = P * self.M / (self.R_star * T)
        return rho

    def sound_speed(self):
        T = atmosphere.temperature(self)
        a = self.a0 * (T / self.T0) ** 0.5
        return a

    def dimensionless_static_pressure(self):
        P = atmosphere.pressure(self)
        delta = P / self.P0
        return delta

    def dimensionless_static_temperature(self):
        T = atmosphere.temperature(self)
        theta = T / self.T0
        return theta

    def dimensionless_static_density(self):
        rho = atmosphere.density(self)
        sigma = rho / self.rho0
        return sigma


if __name__ == '__main__':
    nn = 1000

    h = np.linspace(0, 86000, nn)
    T = np.zeros(nn)
    P = np.zeros(nn)
    rho = np.zeros(nn)
    a = np.zeros(nn)

    for i in range(nn):
        prob = atmosphere(geometric_altitude=h[i])
        T[i] = prob.temperature()
        P[i] = prob.pressure()
        rho[i] = prob.density()
        a[i] = prob.sound_speed()

    fig, ax = plt.subplots(1, 4, figsize=(12, 10))
    st = fig.suptitle("U.S. STANDARD ATMOSPHERE (1976)", fontsize=20)

    ax[0].plot(T, h / 1000, 'b-', linewidth=1.5, label='cruise')
    ax[0].set_xlabel('Temperature: T (K)')
    ax[0].set_ylabel('Altitude: h (km)')

    ax[1].plot(a, h / 1000, 'y-', linewidth=1.5, label='turn')
    ax[1].set_xlabel('Sound Speed: a (m/s)')

    ax[2].plot(P, h / 1000, 'k-', linewidth=1.5, label='constant speed climb')
    ax[2].set_xlabel('Pressure: P (N/${m^2}$)')

    ax[3].plot(rho, h / 1000, 'g-', linewidth=1.5, label='turn')
    ax[3].set_xlabel('Density: rho (Kg/${m^3}$)')

    # shift subplots down:
    st.set_y(0.96)
    fig.subplots_adjust(top=0.90)
    plt.show()