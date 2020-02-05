import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm


def plot_3d(fn, x1_low, x1_high, x2_low, x2_high, stepsize=0.1):
    # Create 2d raster
    x1_steps = np.arange(x1_low, x1_high, stepsize)
    x2_steps = np.arange(x2_low, x2_high, stepsize)
    x1, x2 = np.meshgrid(x1_steps, x2_steps)
    
    # Plot
    y = fn(np.stack((x1, x2), axis=-1))
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x1, x2, y, cmap=cm.plasma, linewidth=0, antialiased=False)
    plt.show()

def rosenbrock(X, a=0, b=1000):
    x1 = X[..., 0]
    x2 = X[..., 1]
    return (a - x1) ** 2 + b * (x2 - x1 ** 2) ** 2

def rastrigin(X, a=1):
    res = X ** 2 - a * np.cos(2 * np.pi * X)
    return 2 * a + np.sum(res, axis=-1)

if __name__ == "__main__":
    plot_3d(rosenbrock, -2, 2, -2, 2)
    plot_3d(rastrigin, -2, 2, -2, 2)
