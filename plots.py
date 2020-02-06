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


def visualize_heatmap(fn, history, extent):
    # heatmap version
    buffer = []
    for i, state in enumerate(history):
        if i % 5 != 0:
            continue
        plt.close("all")
        fig = plt.figure()
        X = np.arange(extent[0], extent[1], 0.1)
        Y = np.arange(extent[2], extent[3], 0.1)
        X_grid, Y_grid = np.meshgrid(X, Y)
        Z = fn(X_grid, Y_grid)
        heat = plt.imshow(Z, extent=extent, cmap='hot')
        # heat = sns.heatmap(Z)
        # heat.invert_yaxis()

        # visualize particles
        x_points = [i["pos"][0] for i in state]
        y_points = [i["pos"][1] for i in state]
        z_points = [i["fit"] for i in state]
        sns.scatterplot(x=x_points, y=y_points)

        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        buffer.append(image)

    imageio.mimsave("particles.gif", buffer, )


def visualize_3D(fn, history):
    # TODO: this whole thing about the plot is still not quite right
    # (0 point is different)
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
        y_points = [i["pos"][1] for i in state]
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
