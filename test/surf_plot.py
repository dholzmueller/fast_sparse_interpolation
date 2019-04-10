import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os.path
import math


def readFromFile(filename):
    if not os.path.isfile(filename):
        return ''

    file = open(filename, 'r')
    result = file.read()
    file.close()
    return result

datastr = readFromFile("../performance_data.csv")
rows = datastr.split("\n")[:-1]
values = np.array([[float(s) for s in r.split(', ')] for r in rows])
# print(values)

dimensions = values[:, 0]
bounds = values[:, 1]
points = values[:, 2]
times = values[:, 3]

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
surf = ax.plot_trisurf(np.log2(dimensions), np.log2(points), np.log10(times), cmap='viridis', edgecolor='none')
plt.show()


