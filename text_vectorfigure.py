import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

X, Y = np.meshgrid(np.arange(0, 1, .05), np.arange(0, 1, .05))
U = np.pi
V = 0

plt.figure()
plt.title('velocity')
Q = plt.quiver(X, Y, U, V, units='width')
plt.show()
