# author: Bao Li # 
# Georgia Institute of Technology #
import sys
import os

sys.path.insert(0, os.getcwd())

import numpy as np
import Sizing_Method.Other.US_Standard_Atmosphere_1976 as atm
import Sizing_Method.Aerodynamics.ThrustLapse as thrust_lapse

# ΑΒΓΔ ΕΖΗΘ ΙΚΛΜ ΝΞΟΠ ΡΣΤΥ ΦΧΨΩ αβγδ εζηθ ικλμ νξοπ ρςτυ φχψω
a = np.linspace(1, 10, 100)
# print(a[0])

b = np.zeros(2)
# print(b)

c = [(1,2,3), (5,6,7)]
# print(c[1][1])
# for i, element in enumerate(c):
   #  for j, element in enumerate(c[i]):
        # print(element)


input_list = [[20, 0], [100, 1000], [300, 2000]]

velocity, altitude = [], []
for i, element in enumerate(input_list):
    # print(i, element)
    velocity.append(element[0])

# print(velocity)
# print(np.zeros((3, 10)))
# print(np.zeros((3, 10))[0,:])
# print("len",len(c))

ind = np.arange(10)
# print("ind", ind)

b = [1, 2, 3, 4, 5, 6, 7]
# print(b[:2])
b = np.append(b, 1)
# print(b)

h = 5000
a = 0.65
v = a * atm.atmosphere(h).sound_speed()
print(v)


thrust_lapse1 = thrust_lapse.thrust_lapse_calculation(altitude=2438, velocity=77).high_bypass_ratio_turbofan()
print(thrust_lapse1)


load_factor = (1 + ((3 * np.pi / 180) * 235 / 9.80665) ** 2) ** 0.5
print(load_factor)

a = 1.78*(1-0.045*10.3**0.68) - 0.64
print(a)

constrains = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

print(constrains.shape[0])


class F:
    def __init__(self, a):
        self.a = a

    def f1(self,b):
        return (self.a * 1+b)

    def f2(self, b):
        return (self.a * 2+b)

    def f3(self,b):
        return (self.a * 3+b)

    allFuncs = [f1, f2, f3]

def main():
    a = 10
    myF = F(a)

    for f in myF.allFuncs:
        #print(f(myF))
        print(myF.allFuncs[1](myF, b=1))

main()

a = [1,2,3,4,5]
print(a[-1])


a = np.array()