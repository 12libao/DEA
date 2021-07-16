# author: Bao Li # 
# Georgia Institute of Technology #
"""Reference:

1: (2.3.1) Mattingly, Jack D., William H. Heiser, and David T. Pratt. Aircraft engine design. American Institute of Aeronautics and Astronautics, 2002.
2. Wedderspoon, J. R. "The high lift development of the A320 aircraft." International Congress of the Aeronautical Sciences, Paper. Vol. 2. No. 2. 1986.
"""

import numpy as np
import Sizing_Method.Aerodynamics.MachNmuber as Ma
import Sizing_Method.Other.US_Standard_Atmosphere_1976 as atm


class aerodynamics_without_pd:
    """this is the class to generate aerodynamics model without dp based on Mattingly Equ 2.9 and section 2.3.1

    1. SI Units
    2. All assumptions, data, tables, and figures are based on large cargo and passenger aircraft,
        where we use A320neo as the baseline.
    """

    def __init__(self, altitude, velocity, AR=10.3):
        """

        :input h (m): altitude
               v (m/s): velocity
               AR: wing aspect ratio, normally between 7 and 10

        :output K1: 2nd Order Coefficient for Cd
                K2: 1st Order Coefficient for Cd
                CD_0: drag coefficient at zero lift
        """

        self.v = velocity
        self.h = altitude

        h = 2.43  # height of winglets
        b = 35.8
        self.AR = AR * (1 + 1.9 * h / b)  # equation 9-88, If the wing has winglets the aspect ratio should be corrected

        # Mach number based on different altitude
        # The Mach number is between 0 to 0.82
        self.a = Ma.mach(self.h, self.v).mach_number()

        if self.a > 0.82:
            print("The Mach number is larger than 0.82, something going wrong!")

        self.CL_min = 0.1  # Assume constant: for most large cargo and passenger, 0.1 < Cl_min < 0.3
        self.CD_min = 0.02  # Assume constant: From Mattingly Figure 2.9

        e = 0.75  # wing planform efficiency factor is between 0.75 and 0.85, no more than 1
        self.K_apo1 = 1 / (np.pi * self.AR * e)  # self.K_apo1 = 1 / (np.pi * self.AR * e)  #

        # K_apo2 is between 0.001 to 0.03 for most large cargo and passenger aircraft
        # Increase with Mach number increase. Thus, assume they have linear relationship
        K_apo2_max = 0.028
        K_apo2_min = 0.001

        a_max = 0.82
        a_min = 0.001

        slop = (K_apo2_max - K_apo2_min) / (a_max - a_min)
        # is the viscous drag due to lift (skin friction and pressure drag)
        self.K_apo2 = K_apo2_max - ((a_max - self.a) * slop)
        # K_apo2 = 0.03*self.a**0.5
        # K_apo2 = 0.015*np.log(self.a)+0.03
        # K_apo2 = 0.0345*self.a**0.25-0.005

    def K1(self):
        """2nd Order Coefficient for Cd"""
        return self.K_apo1 + self.K_apo2

    def K2(self):
        """1st Order Coefficient for Cd"""
        return -2 * self.K_apo2 * self.CL_min

    def CD_0(self):
        """drag coefficient at zero lift"""
        return self.CD_min + self.K_apo2 * self.CL_min ** 2


class aerodynamics_with_pd:
    """Estimation of ΔCL and ΔCD"""

    def __init__(self, altitude, velocity, Hp, n, W_S, P_W=90, sweep_angle=25.0,
                 S=124.0, b=35.8, delta_b=0.5, delta_Dp=0.1, xp=0.5, beta=0.5, AOA_p=0.0, Cf=0.009):
        """

        delta_b = 0.64  # delta_b small then, pd length small
        beta = 0.6 or 0.8

        :param Hp: P_motor/P_total
        :param n: number of motor
        :param P_W:
        :param W_S:
        :param S: wing area
        :param b: wingspan
        :param delta_b:
        :param delta_Dp:
        :param xp:
        :param beta: slipstream correction factor:0-1
        :param CL: lift coefficient
        :param AOA_p: propeller angle of attack
        :param Cf: skin friction coefficient

        :output: 1. ΔCD_0: zero lift drag coefficient changes because of the population distribution
                 2. ΔCL: lift coefficient changes because of the population distribution
        """

        self.h = altitude
        self.v = velocity
        self.n = n
        self.s = S
        self.delta_y1 = delta_b
        self.delta_y2 = delta_Dp
        self.beta = beta
        self.sp = sweep_angle * np.pi / 180
        self.aoa_p = AOA_p
        self.cf = Cf

        self.ar = b ** 2 / self.s  # aspect ratio

        # the diameter of the propulsion, reference 1: Equation 21
        dp = self.delta_y1 * b / (self.n * (1 + self.delta_y2))

        # defining a parameter that indicates how much propulsion-disk
        # area is needed per unit of aircraft weight
        # reference 1: Equation 22
        dp2w = self.delta_y1 ** 2 / (self.n * (1 + self.delta_y2)) ** 2 * self.ar / W_S

        self.rho = atm.atmosphere(geometric_altitude=self.h).density()
        self.t_w = P_W / self.v

        # thrust coefficient Tc of the DP propulsion
        # reference 1: Equation 24
        tc = 1 / self.n * Hp * self.t_w / (self.rho * self.v ** 2 * dp2w)

        # Actuator disk theory shows that there is a maximum theoretical propulsive efficiency
        # for a given thrust coefficient
        ndp_isolated = 0.76
        tc_max = np.pi / 8 * ((2 / ndp_isolated - 1) ** 2 - 1)

        if tc >= tc_max:
            tc = tc_max

        # axial induction factor at the propeller disk (ap) as a
        # function of the propeller thrust coefficient, from the actuator disk theory:
        # reference 1: Equation 25
        ap = 0.5 * ((1 + 8 / np.pi * tc) ** 0.5 - 1)

        # the contraction ratio of the slipstream at the wing
        # leading edge (Rw/RP) can be expressed as
        # reference 1: Equation 26
        rw_rp = ((1 + ap) / (
                1 + ap * (1 + 2 * xp / dp) / ((2 * xp / dp) ** 2 + 1) ** 0.5)) ** 0.5

        # from conservation of mass in incompressible flow: axial induction factor
        self.aw = (ap + 1) / rw_rp ** 2 - 1
        self.m = Ma.mach(self.h, self.v).mach_number()  # Mach number

    def delta_lift_coefficient(self, CL):
        """estimate the lift coefficient changes because of pd"""

        aoa_w = (CL / (2 * np.pi * self.ar)) * (2 + (
                self.ar ** 2 * (1 - self.m ** 2) * (1 + (np.tan(self.sp)) ** 2 / (1 - self.m ** 2)) + 4) ** 0.5)

        delta_cl = 2 * np.pi * ((np.sin(aoa_w) - self.aw * self.beta * np.sin(self.aoa_p - aoa_w))
                                * ((self.aw * self.beta) ** 2 + 2 * self.aw * self.beta * np.cos(self.aoa_p) + 1) ** 0.5
                                - np.sin(aoa_w))
        delta_cl = delta_cl * self.delta_y1
        return delta_cl

    def delta_CD_0(self):
        """estimate the zero lift drag coefficient changes because of the population distribution"""

        delta_cd0 = self.delta_y1 * self.aw ** 2 * self.cf
        return delta_cd0
