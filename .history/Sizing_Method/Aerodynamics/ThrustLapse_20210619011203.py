# author: Bao Li # 
# Georgia Institute of Technology #
"""reference

1. Mattingly, Jack D., William H. Heiser, and David T. Pratt. Aircraft engine design. American Institute of Aeronautics and Astronautics, 2002.
"""
import sys
import os

sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib.pylab as plt
import Sizing_Method.Other.US_Standard_Atmosphere_1976 as atm
import Sizing_Method.Aerodynamics.MachNmuber as Mach


class thrust_lapse_calculation:
    """Estimation for installation losses

    1. The following algebraic equations for installed engine thrust lapse
        are based on the expected performance of advanced engines
    """

    def __init__(self, altitude, velocity):
        """

        :input h (m): altitude
               v (m/s): velocity


        :output thrust_lapse (alpha): thrust lapse ratio
        """

        self.h = altitude
        self.v = velocity

        self.gamma = 1.4  # Ratio of Specific Heats: atmospheric air ( gamma = 1.4)
        self.atoms = atm.atmosphere(self.h)
        self.M = Mach.mach(self.h, self.v).mach_number()

        self.theta = self.atoms.dimensionless_static_temperature()  # dimensionless static temperature
        self.delta = self.atoms.dimensionless_static_pressure()  # dimensionless static pressure

        self.theta_0_break = 1.072  # The Theta Break: assume = 1.0 (from reference 1: Mattingly)
        self.TR = self.theta_0_break  # The Throttle Ratio (from Equation D.6 reference 1: Mattingly)

        self.theta_0 = self.theta * (1 + 0.5 * (self.gamma - 1) * self.M ** 2)
        self.delta_0 = self.delta * (1 + 0.5 * (self.gamma - 1) * self.M ** 2) ** (self.gamma / (self.gamma - 1))

    def high_bypass_ratio_turbofan(self):
        """the engine for A320neo is a high bypass ratio turbofan

        from Equation 2.53 reference 1: Mattingly
        """

        if self.theta_0 <= self.TR:
            thrust_lapse = self.delta_0 * (1 - 0.49 * self.M ** 0.5)
        else:
            thrust_lapse = self.delta_0 * (1 - 0.49 * self.M ** 0.5 - 3 * (self.theta_0 - self.TR) / (1.5 + self.M))

        return thrust_lapse


if __name__ == '__main__':
    nn = 100
    h = [0, 500, 3000, 6000, 9000, 12000]
    v = np.linspace(0, 340, nn)
    m = np.zeros([len(h), nn])
    thrust_lapse = np.zeros([len(h), nn])

    for i in range(len(h)):
        for j in range(nn):
            m[i, j] = Mach.mach(h[i], v[j]).mach_number()
            thrust_lapse[i, j] = thrust_lapse_calculation(h[i], v[j]).high_bypass_ratio_turbofan()

    plt.figure(figsize=(8, 6))

    for i in range(len(h)):
        plt.plot(m[i, :], thrust_lapse[i, :], label=h[i])

    plt.xlabel('Mach Number')
    plt.ylabel('Thrust Ratio, T/$T_{SL}$')
    plt.title('High Bypass Turbofan Thrust Ratio versus Mach Number \n'
              'Throttle Ratio = 1.072, ISA. \n'
              'Altitude unit: meter')
    plt.xlim(0, 1)
    plt.ylim(0, 1.2)
    plt.legend(loc=0)
    plt.grid()
    plt.show()
