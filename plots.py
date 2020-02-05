import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm


### Rosenbrock
def plt_rosenbrock():
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


def rastrigin(*X, **kwargs):
    A = kwargs.get('A', 10)
    return A*len(X) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])

### Rastrigin
def plt_rastrigin():
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    X = np.arange(-2, 2, 0.1)
    Y = np.arange(-2, 2, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = rastrigin(X, Y, A=1)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0,
    antialiased=False)
    plt.show()


if __name__ == "__main__":
    plt_rosenbrock()
    plt_rastrigin()    


