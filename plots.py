import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import utils
import seaborn as sns
import imageio


def plot_3d(fn, x1_low, x1_high, x2_low, x2_high, stepsize=0.1):
    # Create 2d raster
    x1_steps = np.arange(x1_low, x1_high, stepsize)
    x2_steps = np.arange(x2_low, x2_high, stepsize)
    x1, x2 = np.meshgrid(x1_steps, x2_steps)
    
    # Plot
    y = fn(x1, x2)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x1, x2, y, cmap=cm.plasma, linewidth=0, antialiased=False)
    plt.show()


def visualize_heatmap(fn, history):
    # heatmap version
    buffer = []
    for state in history:
        plt.close("all")
        fig = plt.figure()
        X = np.arange(-2, 2, 0.1)
        Y = np.arange(-2, 2, 0.1)
        X_grid, Y_grid = np.meshgrid(X, Y)
        Z = fn(X_grid, Y_grid)
        sns.heatmap(Z)

        # visualize particles
        x_points = [i["pos"][0] for i in state]
        y_points = [i["pos"][0] for i in state]
        z_points = [i["fit"] for i in state]
        sns.scatterplot(x=x_points, y=y_points)


        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        buffer.append(image)

    imageio.mimsave("particles.gif", buffer, )


def visualize_3D(fn, history):
    buffer = []
    for state in history:
        plt.close("all")
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        X = np.arange(-2, 2, 0.1)
        Y = np.arange(-2, 2, 0.1)
        X, Y = np.meshgrid(X, Y)
        a = 0
        b = 1000
        Z = utils.rastrigin(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0,
        antialiased=False)

        # visualize particles
        x_points = [i["pos"][0] for i in state]
        y_points = [i["pos"][0] for i in state]
        z_points = [i["fit"] for i in state]
        ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv')

        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        buffer.append(image)

    imageio.mimsave("particles.gif", buffer, )


if __name__ == "__main__":
    plot_3d(utils.rosenbrock, -2, 2, -2, 2)
    plot_3d(utils.rastrigin, -2, 2, -2, 2)
