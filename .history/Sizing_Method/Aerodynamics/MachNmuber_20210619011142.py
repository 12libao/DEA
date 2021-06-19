# author: Bao Li # 
# Georgia Institute of Technology #
import sys
import os

sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib.pylab as plt
import Sizing_Method.Other.US_Standard_Atmosphere_1976 as atm


class mach:
    """Mach Number"""

    def __init__(self, altitude, velocity):
        """

        :input h (m): altitude
               v (m/s): velocity

        :output a: Mach number
        """
        self.h = altitude
        self.v = velocity
        self.atoms = atm.atmosphere(self.h)

    def mach_number(self):
        a = self.v / self.atoms.sound_speed()
        return a
