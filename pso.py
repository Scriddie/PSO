"""
initialize swarm, update location on each iteration
"""
import numpy as np
from copy import deepcopy
from plots import rastrigin
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio

best_fitness = np.inf
best_position = np.array([0, 0])

def initialize(n, x_min, x_max, y_min, y_max):
    particles = [
        {
            "pos": np.array([np.random.uniform(x_min, x_max),
                             np.random.uniform(y_min, y_max)]),
            "v": np.array([np.random.uniform(0, 0.1),
                           np.random.uniform(0, 0.1)]),
            "fit": np.inf
        }
        for i in range(n)
    ]
    for p in particles:
        p["best_pos"] = (p["pos"])
    return particles

def update(particles, fitness_func):
    global best_fitness
    global best_position
    update = []
    for p in particles:
        r = np.random.uniform(0, 1)
        v = (
            p["v"] +
            r * (p["pos"] - p["best_pos"]) +
            r * (p["pos"] - best_position)
        )
        update.append(v)
    for i, p in enumerate(particles):
        p["pos"] += update[i]
        p["fit"] = fitness_func(p["pos"][0], p["pos"][1])
        if p["fit"] > best_fitness:
            best_fitness = p["fit"]
            best_position = p["pos"]
    return particles

def train(num_particles, num_iter, fitness_func):
    particles = initialize(num_particles, -2, 2, -2, 2)
    history = []
    for i in range(num_iter):
        particles = update(particles, fitness_func)
        history.append(deepcopy(particles))
    return history


if __name__ == "__main__":

    # TODO: check if this works, numbers in history look good at first sight but gif doesnt do anything
    history = train(10, 50, rastrigin)

    buffer = []
    for state in history:
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        X = np.arange(-2, 2, 0.1)
        Y = np.arange(-2, 2, 0.1)
        X, Y = np.meshgrid(X, Y)
        a = 0
        b = 1000
        Z = rastrigin(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0,
        antialiased=False)

        # visualize particles
        x_points = [i["pos"][0] for i in history[0]]
        y_points = [i["pos"][0] for i in history[0]]
        z_points = [i["fit"] for i in history[0]]
        ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');

        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        buffer.append(image)

    imageio.mimsave("particles.gif", buffer, )


