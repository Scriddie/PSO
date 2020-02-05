from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


### Rosenbrock function

fig = plt.figure()
ax = fig.gca(projection="3d")
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
a = 0
b = 1000
Z = np.add(
    np.square(np.subtract(a, X)),
    np.multiply(b, np.square(np.subtract(Y, np.square(X))))
)
surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0,
 antialiased=False)
plt.show()


### Rastrigin function

def rastrigin(*X, **kwargs):
    A = kwargs.get('A', 10)
    return A*len(X) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])

fig = plt.figure()
ax = fig.gca(projection="3d")
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = rastrigin(X, Y, A=1)
surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0,
 antialiased=False)
plt.show()


