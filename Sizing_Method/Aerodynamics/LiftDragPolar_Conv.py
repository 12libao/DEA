# author: Bao Li # 
# Georgia Institute of Technology #
"""Reference:

1: (2.3.1) Mattingly, Jack D., William H. Heiser, and David T. Pratt. Aircraft engine design. American Institute of Aeronautics and Astronautics, 2002.
2. Wedderspoon, J. R. "The high lift development of the A320 aircraft." International Congress of the Aeronautical Sciences, Paper. Vol. 2. No. 2. 1986.
"""

import numpy as np
import matplotlib.pylab as plt
import Sizing_Method.Other.US_Standard_Atmosphere_1976 as atm


class lift_drag_polar:
    """this is the class to generate lift-drag polar equation based on Mattingly Equ 2.9 and section 2.3.1

    1. SI Units
    2. All assumptions, data, tables, and figures are based on large cargo and passenger aircraft,
        where we use A320neo as the baseline.
    """

    def __init__(self, velocity, altitude, AR=10.3):
        """

        :input v (m/s): velocity
               h (m): altitude
               AR: wing aspect ratio, normally between 7 and 10

        :output K1: 2nd Order Coefficient for Cd
                K2: 1st Order Coefficient for Cd
                CD_0: drag coefficient at zero lift
        """

        self.v = velocity
        self.h = altitude
        self.AR = AR

        # Mach number based on different altitude
        # The Mach number is between 0 to 0.82
        self.atoms = atm.atmosphere(self.h)
        self.a = self.v / self.atoms.sound_speed()
        if self.a > 0.85:
            print("The Mach number is larger than 0.85, something going wrong!")

        self.CL_min = (0.1 + 0.3) / 2  # Assume constant: for most large cargo and passenger, 0.1 < Cl_min < 0.3
        self.CD_min = 0.018  # Assume constant: From Mattingly Figure 2.9

    def K_apo1(self):
        """is the inviscid drag due to lift (induced drag)

        :param AR: wing aspect ratio, normally between 7 and 10
        """
        e = (0.75 + 0.85) / 2  # wing planform efficiency factor is between 0.75 and 0.85, no more than 1
        return 1 / (np.pi * self.AR * e)

    def K_apo2(self):
        """is the viscous drag due to lift (skin friction and pressure drag)

        K_apo2 is between 0.001 to 0.03 for most large cargo and passenger aircraft
        Increase with Mach number increase
        Thus, assume they have linear relationship
        """
        K_apo2_max = 0.028
        K_apo2_min = 0.001

        a_max = 0.85
        a_min = 0.001

        slop = (K_apo2_max - K_apo2_min) / (a_max - a_min)
        K_apo2 = K_apo2_max - ((a_max - self.a) * slop)
        return K_apo2

    def K1(self):
        """2nd Order Coefficient for Cd"""
        return lift_drag_polar.K_apo1(self) + lift_drag_polar.K_apo2(self)

    def K2(self):
        """1st Order Coefficient for Cd"""
        return -2 * lift_drag_polar.K_apo2(self) * self.CL_min

    def CD_0(self):
        """drag coefficient at zero lift"""
        return self.CD_min + lift_drag_polar.K_apo2(self) * self.CL_min ** 2

    def lift_drag_polar_equation(self, CL):
        CD = lift_drag_polar.K1(self) * CL ** 2 + lift_drag_polar.K2(self) * CL + lift_drag_polar.CD_0(self)
        inviscid_drag = lift_drag_polar.K_apo1(self) * CL ** 2
        viscous_drag = lift_drag_polar.K_apo2(self) * CL ** 2 - 2 * lift_drag_polar.K_apo2(self) * self.CL_min * CL
        parasite_drag = lift_drag_polar.CD_0(self)
        return CD, inviscid_drag, viscous_drag, parasite_drag

    def maximum_lift_coefficient(self, sweep_angle):
        """Based on reference 1: table 2.1, and reference 2"""
        CL_max_takeoff = 2.3 * np.cos(sweep_angle)
        CL_max_landing = 2.8 * np.cos(sweep_angle)
        return CL_max_takeoff, CL_max_landing


if __name__ == '__main__':
    AR = 10.3
    input_list = [[10, 20], [1000, 100], [12000, 250]]
    n = len(input_list)
    velocity, altitude = [], []
    for i, element in enumerate(input_list):
        altitude.append(element[1])
        velocity.append(element[0])

    nn = 100
    # CL = np.linspace(0.0, 1.0, nn)
    CL = np.linspace(0.0, 0.25, nn)

    CD = np.zeros((n, nn))

    for i, element in enumerate(input_list):
        prob = lift_drag_polar(velocity=element[0], altitude=element[1], AR=AR)
        for j in range(nn):
            CD[i, j], _, _, _ = prob.lift_drag_polar_equation(CL[j])

    plt.figure(figsize=(8, 6))
    plt.plot(CD[0, :], CL, 'b-', linewidth=1.5, label='Takeoff')
    plt.plot(CD[1, :], CL, 'k-', linewidth=1.5, label='Climb')
    plt.plot(CD[2, :], CL, 'g-', linewidth=1.5, label='Cruise')
    plt.xlabel('$C_{D}$')
    plt.ylabel('$C_{L}$')
    plt.title('Lift Drag Polar:Cl=[0, 0.25] \n'
              'Assume Camber Airfoil')
    plt.legend(loc=0)
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 10))
    inviscid_drag = np.zeros(n)
    viscous_drag = np.zeros(n)
    parasite_drag = np.zeros(n)
    lift_drag = np.zeros(n)

    ind = np.arange(n)  # the x locations for the groups
    width = 0.6  # the width of the bars: can also be len(x) sequence
    CL_h = [2, 1.2, 0.6]  # takeoff Cl= Cl_max for takeoff
    for i, element in enumerate(input_list):
        prob = lift_drag_polar(velocity=element[0], altitude=element[1], AR=AR)
        _, inviscid_drag[i], viscous_drag[i], parasite_drag[i] = prob.lift_drag_polar_equation(CL=CL_h[i])
        lift_drag[i] = inviscid_drag[i] + viscous_drag[i]

    p1 = plt.bar(ind, inviscid_drag, width)
    p2 = plt.bar(ind, viscous_drag, width, bottom=inviscid_drag)  # , yerr=womenStd)
    p3 = plt.bar(ind, parasite_drag, width, bottom=lift_drag)

    plt.ylabel('Drag Coefficients')
    plt.title('Drag Breakdown \n'
              'Inviscid Drag: due to lift (induced drag) \n'
              'Viscous Drag: due to lift (skin friction and pressure drag) \n'
              'Most drag at cruise is parasite drag \n'
              'Most drag at takeoff is lift-dependent drag \n')
    plt.xticks(ind, ('Takeoff', 'Climb', 'Cruise'))
    plt.yticks(np.arange(0, 2, 20))
    plt.legend((p1[0], p2[0], p3[0]), ('Inviscid Drag', 'Viscous Drag', 'Zero Lift Drag'))
    plt.show()