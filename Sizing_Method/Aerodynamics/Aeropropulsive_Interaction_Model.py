# author: Bao Li # 
# Georgia Institute of Technology #
"""
Reference:

1. De Vries, Reynard, Malcom Brown, and Roelof Vos. "Preliminary sizing method for hybrid-electric
    distributed-propulsion aircraft." Journal of Aircraft 56.6 (2019): 2172-2188.
2. Patterson, M. D., and German, B. J., “Simplified Aerodynamics Models to Predict the Effects of
    Upstream Propellers on Wing Lift,” 53rd AIAA Aerospace Sciences Meeting, Kissimmee, FL, USA, January 5-9 2015.


A series of delta terms (ΔCL, ΔCD0 , ΔCDi , and Δηdp) must be estimated in order to incorporate the
aero propulsive interaction effects in the design process. This section proposes a method to estimate
these terms for distributed propellers mounted ahead of the wing leading edge.


The method proposed in this section is based on the approach of Patterson and German. It represents the propellers
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

import numpy as np
import matplotlib.pylab as plt
import Sizing_Method.Other.US_Standard_Atmosphere_1976 as atm
import Sizing_Method.Aerodynamics.MachNmuber as Mach
import Sizing_Method.Aerodynamics.LiftDragPolar_Conv as ad


class Aero_propulsion:
    """Estimation of ΔCL and ΔCD"""

    def __init__(self, altitude, velocity, Hp, n, P_W, W_S, sweep_angle=25.0,
                 S=124.0, b=35.8, delta_b=0.5, delta_Dp=0.1, xp=0.5, beta=0.6, AOA_p=0.0, Cf=0.009):
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

        # axial induction factor at the propeller disk (ap) as a
        # function of the propeller thrust coefficient,
        # from the actuator disk theory:
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
        # geometric angle of attack of the wing, where assume cl=1
        self.aoa_w = (1 / (2 * np.pi * self.ar)) * (2 + (self.ar ** 2 * (1 - self.m ** 2) * (1 + (np.tan(self.sp)) ** 2 / (1 - self.m ** 2)) + 4) ** 0.5)

    def delta_lift_coefficient(self):
        """estimate the lift coefficient changes because of the population distribution"""

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
        delta_cl = Aero_propulsion.delta_lift_coefficient(self)
        delta_cd0 = Aero_propulsion.delta_CD_0(self)
        CD = K1 * (CL + delta_cl) ** 2 + K2 * (CL + delta_cl) + (CD_0 + delta_cd0)

        inviscid_drag = K_apo1 * (CL + delta_cl) ** 2
        viscous_drag = K_apo2 * (CL + delta_cl) ** 2 - 2 * K_apo2 * CL_min * (
                    CL + delta_cl)
        parasite_drag = CD_0 + delta_cd0

        return CD, inviscid_drag, viscous_drag, parasite_drag


if __name__ == '__main__':
    AR = 10.3
    input_list = [[10, 20], [1000, 100], [12000, 250]]
    n = len(input_list)
    velocity, altitude = [], []
    for i, element in enumerate(input_list):
        altitude.append(element[0])
        velocity.append(element[1])

    nn = 10
    CL = np.linspace(0.0, 1.0, nn)
    # CL = np.linspace(0.0, 0.25, nn)

    CD = np.zeros((n, nn))

    for i, element in enumerate(input_list):
        prob = Aero_propulsion(altitude=element[0], velocity=element[1], Hp=0.5, n=12, P_W=0.3*element[1], W_S=500)
        for j in range(nn):
            CD[i, j], _, _, _ = prob.lift_drag_polar_equation(CL[j])
            print(j)

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
        prob = Aero_propulsion(altitude=element[0], velocity=element[1], Hp=0.5, n=12, P_W=0.3*element[1], W_S=500)
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