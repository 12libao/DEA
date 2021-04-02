# author: Bao Li # 
# Georgia Institute of Technology #
import numpy as np


a = np.linspace(1, 10, 100)
print(a[0])

b = np.zeros(2)
print(b)

c = [(1,2,3), (5,6,7)]
print(c[1][1])
for i, element in enumerate(c):
    for j, element in enumerate(c[i]):
        print(element)


input_list = [[20, 0], [100, 1000], [300, 2000]]
velocity, altitude = [], []
for i, element in enumerate(input_list):
    # print(i, element)
    velocity.append(element[0])

# print(velocity)
# print(np.zeros((3, 10)))
# print(np.zeros((3, 10))[0,:])
print("len",len(c))

ind = np.arange(10)
print("ind", ind)

print(np.pi)