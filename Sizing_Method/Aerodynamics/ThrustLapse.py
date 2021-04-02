# author: Bao Li # 
# Georgia Institute of Technology #
"""reference

1. Mattingly, Jack D., William H. Heiser, and David T. Pratt. Aircraft engine design. American Institute of Aeronautics and Astronautics, 2002.
"""

import Sizing_Method.Other.US_Standard_Atmosphere_1976 as atm
import Sizing_Method.Aerodynamics.MachNmuber as Mach


class thrust_lapse_calculation:
    """Estimation for installation losses

    1. The following algebraic equations for installed engine thrust lapse
        are based on the expected performance of advanced engines
    2.
    """

    def __init__(self, altitude, velocity):
        """

        :input v (m/s): velocity
               h (m): altitude
               AR: wing aspect ratio, normally between 7 and 10

        :output K1: 2nd Order Coefficient for Cd
                K2: 1st Order Coefficient for Cd
                CD_0: drag coefficient at zero lift
        """

        self.h = altitude
        self.v = velocity

        self.gamma = 1.4  # Ratio of Specific Heats: atmospheric air ( gamma = 1.4)
        self.atoms = atm.atmosphere(self.h)
        self.M = Mach.mach(self.h, self.v).mach_number()

        self.theta = self.atoms.dimensionless_static_temperature()  # dimensionless static temperature
        self.delta = self.atoms.dimensionless_static_pressure()  # dimensionless static pressure

        self.theta_0_break = 1.0  # The Theta Break: assume = 1.0 (from reference 1: Mattingly)
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
    thrust_lapse1 = thrust_lapse_calculation(altitude=10, velocity=10).high_bypass_ratio_turbofan()  # the case for takeoff
    thrust_lapse2 = thrust_lapse_calculation(altitude=10000, velocity=250).high_bypass_ratio_turbofan()  # the case for cruise
    print(thrust_lapse1, thrust_lapse2)