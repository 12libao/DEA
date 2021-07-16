# author: Bao Li # 
# Georgia Institute of Technology #
"""
Reference:

1. De Vries, Reynard, Malcom Brown, and Roelof Vos. "Preliminary sizing method for hybrid-electric
    distributed-propulsion aircraft." Journal of Aircraft 56.6 (2019): 2172-2188.
2. Patterson, M. D., and German, B. J., “Simplified Aerodynamics Models to Predict the Effects of
    Upstream Propellers on Wing Lift,” 53rd AIAA Aerospace Sciences Meeting, Kissimmee, FL, USA, January 5-9 2015.


A series of delta terms (ΔCL, ΔCD0 , ΔCDi) must be estimated in order to incorporate the
aero propulsive interaction effects in the design process. This section proposes a method to estimate
these terms for distributed propellers mounted ahead of the wing leading edge.


The method is based on the approach of Patterson and German (reference 2). It represents the propellers
as actuator disks and the wing as a flat plate, incorporating a semi-empirical correction for finite slipstream height.
The model includes several assumptions worth highlighting:

1. The velocity increase at the actuator disk is computed assuming uniform axial inflow.
2. Variations in lift due to swirl are neglected (actuator disk assumption).
3. The flow over the wing is attached.
4. The airfoil is symmetric, and thus zero lift is produced at a = 0.
5. The effect of each propeller on the adjacent ones is neglected.
6. The effect of the propellers on the wing is limited to the span wise interval occupied by the disks (ΔY/b).
7. Within this span wise interval, the effect on the wing is considered uniform in span wise direction.
8. The wing is supposed to be fully immersed in the slipstream, that is,
half of the slipstream flows under the wing and half over the wing.
9. incompressible flow condition.

"""
import sys
import os

sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib.pylab as plt
import Sizing_Method.Other.US_Standard_Atmosphere_1976 as atm
import Sizing_Method.Aerodynamics.MachNmuber as Mach
import Sizing_Method.Aerodynamics.LiftDragPolar_Conv as ad


class Aero_propulsion:
    """Estimation of ΔCL and ΔCD"""

    def __init__(self, altitude, velocity, Hp, n, P_W, W_S, sweep_angle=25.0,
                 S=124.0, b=35.8, delta_b=0.5, delta_Dp=0.1, xp=0.5, beta=0.5, AOA_p=0.0, Cf=0.009):
        """

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

        :output: 1. delta_CD_0: zero lift drag coefficient changes because of the population distribution
                 2. delta_CL: lift coefficient changes because of the population distribution
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

        print(tc)

        tc = 0.611  # check whether the model is accurate

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

        self.m = Mach.mach(self.h, self.v).mach_number()  # Mach number
        # geometric angle of attack of the wing, where assume cl=1 self.aoa_w = (1 / (2 * np.pi * self.ar)) * (2 + (
        # self.ar ** 2 * (1 - self.m ** 2) * (1 + (np.tan(self.sp)) ** 2 / (1 - self.m ** 2)) + 4) ** 0.5)

        # self.aoa_w = -2 * np.pi / 180  # check whether the model is accurate

    def delta_lift_coefficient(self, CL):
        """estimate the lift coefficient changes because of the population distribution"""
        self.aoa_w = (CL / (2 * np.pi * self.ar)) * (2 + (
                self.ar ** 2 * (1 - self.m ** 2) * (1 + (np.tan(self.sp)) ** 2 / (1 - self.m ** 2)) + 4) ** 0.5)

        delta_cl = 2 * np.pi * ((np.sin(self.aoa_w) - self.aw * self.beta * np.sin(self.aoa_p - self.aoa_w))
                                * ((self.aw * self.beta) ** 2 + 2 * self.aw * self.beta * np.cos(self.aoa_p) + 1) ** 0.5
                                - np.sin(self.aoa_w))
        delta_cl_total = delta_cl * self.delta_y1
        return delta_cl_total

    def delta_CD_0(self):
        """estimate the zero lift drag coefficient changes because of the population distribution"""

        delta_cd0 = self.delta_y1 * self.aw ** 2 * self.cf
        return delta_cd0

    def delta_lift_induced_drag_coefficient(self, CL):
        """estimate the lift induced drag coefficient changes because of the population distribution"""

        e = (0.75 + 0.85) / 2  # wing planform efficiency factor is between 0.75 and 0.85, no more than 1
        delta_cl = Aero_propulsion.delta_lift_coefficient(self)
        delta_cdi = (delta_cl ** 2 + 2 * CL * delta_cl) / (np.pi * self.ar * e)

        return delta_cdi

    def lift_drag_polar_equation(self, CL):
        K1 = ad.lift_drag_polar(velocity=self.v, altitude=self.h).K1()
        K2 = ad.lift_drag_polar(velocity=self.v, altitude=self.h).K2()
        CD_0 = ad.lift_drag_polar(velocity=self.v, altitude=self.h).CD_0()
        K_apo1 = ad.lift_drag_polar(velocity=self.v, altitude=self.h).K_apo1()
        K_apo2 = ad.lift_drag_polar(velocity=self.v, altitude=self.h).K_apo2()

        CL_min = (0.1 + 0.3) / 2  # Assume constant: for most large cargo and passenger, 0.1 < Cl_min < 0.3
        delta_cl = Aero_propulsion.delta_lift_coefficient(self, CL)
        delta_cd0 = Aero_propulsion.delta_CD_0(self)
        CD = K1 * (CL + delta_cl) ** 2 + K2 * (CL + delta_cl) + (CD_0 + delta_cd0)

        inviscid_drag = K_apo1 * (CL + delta_cl) ** 2
        viscous_drag = K_apo2 * (CL + delta_cl) ** 2 - 2 * K_apo2 * CL_min * (
                CL + delta_cl)
        parasite_drag = CD_0 + delta_cd0

        return CD, inviscid_drag, viscous_drag, parasite_drag


if __name__ == '__main__':
    input_list = [[0, 62], [3000, 150], [12000, 250]]
    n = len(input_list)
    velocity, altitude = [], []
    for i, element in enumerate(input_list):
        altitude.append(element[0])
        velocity.append(element[1])

    nn = 100
    CL = np.linspace(0.0, 2.3, nn)
    # CL = np.linspace(0.0, 0.25, nn)

    CD = np.zeros((2 * n, nn))

    for i, element in enumerate(input_list):
        prob = Aero_propulsion(altitude=element[0], velocity=element[1], Hp=0.5, n=12, P_W=0.3 * element[1], W_S=5600)
        prob2 = ad.lift_drag_polar(altitude=element[0], velocity=element[1])
        for j in range(nn):
            CD[i, j], _, _, _ = prob.lift_drag_polar_equation(CL=CL[j])
            CD[i + n, j], _, _, _ = prob2.lift_drag_polar_equation(CL=CL[j])

    plt.figure(figsize=(8, 6))
    plt.plot(CD[0, :], CL, 'b-', linewidth=1.5, label='Takeoff with DP')
    plt.plot(CD[1, :], CL, 'k-', linewidth=1.5, label='Climb with DP')
    plt.plot(CD[2, :], CL, 'g-', linewidth=1.5, label='Cruise with DP')
    plt.plot(CD[3, :], CL, 'b--', linewidth=1.5, label='Takeoff')
    plt.plot(CD[4, :], CL, 'k--', linewidth=1.5, label='Climb')
    plt.plot(CD[5, :], CL, 'g--', linewidth=1.5, label='Cruise')
    plt.xlabel('$C_{D}$')
    plt.ylabel('$C_{L}$')
    plt.title('Lift Drag Polar:Cl=[0, 1]')
    plt.legend(loc=0)
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    inviscid_drag, inviscid_drag2 = np.zeros(n), np.zeros(n)
    viscous_drag, viscous_drag2 = np.zeros(n), np.zeros(n)
    parasite_drag, parasite_drag2 = np.zeros(n), np.zeros(n)
    lift_drag, lift_drag2 = np.zeros(n), np.zeros(n)

    ind = np.arange(n)  # the x locations for the groups
    width = 0.3  # the width of the bars: can also be len(x) sequence
    CL_h = [2, 1.2, 0.6]  # takeoff Cl= Cl_max for takeoff
    for i, element in enumerate(input_list):
        prob = Aero_propulsion(altitude=element[0], velocity=element[1], Hp=0.5, n=12, P_W=0.3 * element[1], W_S=5600)
        prob2 = ad.lift_drag_polar(altitude=element[0], velocity=element[1])
        _, inviscid_drag[i], viscous_drag[i], parasite_drag[i] = prob.lift_drag_polar_equation(CL=CL_h[i])
        _, inviscid_drag2[i], viscous_drag2[i], parasite_drag2[i] = prob2.lift_drag_polar_equation(CL=CL_h[i])
        lift_drag[i] = inviscid_drag[i] + viscous_drag[i]
        lift_drag2[i] = inviscid_drag2[i] + viscous_drag2[i]

    p1 = plt.bar(ind - width / 2, inviscid_drag2, width, color='b', alpha=0.8, label='inviscid drag')
    p2 = plt.bar(ind - width / 2, viscous_drag2, width, bottom=inviscid_drag2, color='g', alpha=0.8,
                 label='viscous drag')
    p3 = plt.bar(ind - width / 2, parasite_drag2, width, bottom=lift_drag2, color='y', alpha=0.8,
                 label='zero_lift drag')

    p4 = plt.bar(ind + width / 2, inviscid_drag, width, color='b', alpha=0.5, label='inviscid drag with DP')
    p5 = plt.bar(ind + width / 2, viscous_drag, width, bottom=inviscid_drag, color='g', alpha=0.5,
                 label='viscous drag with DP')
    p6 = plt.bar(ind + width / 2, parasite_drag, width, bottom=lift_drag, color='y', alpha=0.5,
                 label='zero_lift drag with DP')

    plt.ylabel('Drag Coefficients')
    plt.title('Drag Breakdown without DP vs with DP')
    plt.xticks(ind, ('Takeoff', 'Climb', 'Cruise'))
    plt.yticks(np.arange(0, 2, 20))
    plt.legend(loc=0)
    plt.show()

    CL = np.linspace(1.5, 2.7, 20)
    v = 62.0
    h = 0.0

    prob = Aero_propulsion(altitude=h, velocity=v, Hp=0.5, n=12, P_W=0.3 * v, W_S=5600, delta_b=0.64)
    K1 = ad.lift_drag_polar(velocity=v, altitude=h).K1()
    K2 = ad.lift_drag_polar(velocity=v, altitude=h).K2()
    CD_0 = ad.lift_drag_polar(velocity=v, altitude=h).CD_0()

    delta_cd0 = prob.delta_CD_0()

    delta_cl = np.zeros(len(CL))
    delta_cd = np.zeros(len(CL))

    for i in range(len(CL)):
        delta_cl[i] = prob.delta_lift_coefficient(CL=CL[i])
        CD1 = K1 * (CL[i] + delta_cl[i]) ** 2 + K2 * (CL[i] + delta_cl[i]) + (CD_0 + delta_cd0)
        CD2 = K1 * CL[i] ** 2 + K2 * CL[i] + CD_0
        delta_cd[i] = CD1 - CD2 + delta_cd0

    plt.figure(figsize=(8, 6))
    plt.plot(CL, delta_cl, 'b-', linewidth=1.5, label='delta $C_{L}$')
    plt.plot(CL, delta_cd, 'k-', linewidth=1.5, label='delta $C_{D}$')
    plt.xlabel('Airframe lift coefficient $C_{L}$')
    plt.ylabel('Lift or drag coefficient increase $C_{L}$, $C_{D}$')
    plt.title('Distributed Propulsion $T_{c}$=0.611')
    plt.legend(loc=0)
    plt.grid()
    plt.show()
