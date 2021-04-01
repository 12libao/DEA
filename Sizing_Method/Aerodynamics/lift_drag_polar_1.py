# author: Bao Li # 
# Georgia Institute of Technology #
"""Reference:

1: (2.3.1) Mattingly, Jack D., William H. Heiser, and David T. Pratt. Aircraft engine design. American Institute of Aeronautics and Astronautics, 2002.
"""

import numpy as np
import matplotlib.pylab as plt
import Sizing_Method.US_Standard_Atmosphere_1976 as atm


class lift_drag_polar:
    """this is the class to generate lift-drag polar equation based on Mattingly Equ 2.9 and section 2.3.1

    1. SI Units
    2. All assumptions, data, tables, and figures are based on large cargo and passenger aircraft,
        where we use A320neo as the baseline.
    """

    def __init__(self, velocity, altitude):
        """

        :input v (m/s): velocity
               h (m): altitude

        :output P (Pa = N/m^2): pressure
                T (K): temperature
                rho (kg/m^3): density
                a (m/s): sound speed
        """

        self.v = velocity
        self.h = altitude

        # Mach number based on different altitude
        # The Mach number is between 0 to 0.82
        self.atoms = atm.atmosphere(self.h)
        self.a = self.v / self.atoms.sound_speed()
        if self.a > 0.85:
            print("The Mach number is larger than 0.85, something going wrong!")

        self.CL_min = (0.1 + 0.3) / 2  # Assume constant: for most large cargo and passenger, 0.1 < Cl_min < 0.3
        self.CD_min = 0.018  # Assume constant: From Mattingly Figure 2.9

    def K_apo1(self, AR):
        """is the inviscid drag due to lift (induced drag)

        :param AR: wing aspect ratio, normally between 7 and 10
        """
        e = (0.75 + 0.85) / 2  # wing planform efficiency factor is between 0.75 and 0.85, no more than 1
        return 1 / (np.pi * AR * e)

    def K_apo2(self):
        """is the viscous drag due to lift (skin friction and pressure drag)

        K_apo2 is between 0.001 to 0.03 for most large cargo and passenger aircraft
        Increase with Mach number increase
        Thus, assume they have linear relationship
        """
        slop = (0.028 - 0.002) / (0.82 - 0.01)
        return self.a * slop

    def K1(self):
        return lift_drag_polar.K_apo2(self) + lift_drag_polar.K_apo2(self)

    def K2(self):
        return -2 * lift_drag_polar.K_apo2(self) * self.CL_min

    def CD_0(self):
        return self.CD_min + lift_drag_polar.K_apo2(self) * self.CL_min ** 2

    def lift_drag_polar_equation(self, CL):
        CD = lift_drag_polar.K1(self)*CL**2 + lift_drag_polar.K2(self) + lift_drag_polar.CD_0(self)
        return CD




























if __name__ == '__main__':
    prob = lift_drag_polar(velocity=340, altitude=1000)

























































