"""
initialize swarm, update location on each iteration
"""
import numpy as np
from copy import deepcopy
import plots
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import os
from scipy import optimize
from importlib import reload
reload(plots)
reload(utils)

best_fitness = np.inf
best_fit_hist = [deepcopy(best_fitness)]
best_position = np.array([0, 0])
best_pos_hist = [deepcopy(best_position)]

def initialize(fn, n, x_min, x_max, y_min, y_max):
    global best_fitness
    global best_position
    particles = [
        {
            "pos": np.array([np.random.uniform(x_min, x_max),
                             np.random.uniform(y_min, y_max)]),
            "v": np.array([np.random.uniform(0, 0.1),
                           np.random.uniform(0, 0.1)]),
        }
        for i in range(n)
    ]
    # TODO: this doesnt seem to work
    for p in particles:
        p["best_pos"] = deepcopy(p["pos"])
        p["fit"] = deepcopy(fn(p["pos"]))
        p["best_fit"] = deepcopy(p["fit"])
        if p["fit"] < best_fitness:
            best_fitness = deepcopy(p["fit"])
            best_position = deepcopy(p["pos"])
    return particles


def update(fn, particles, a, b, c):
    global best_fitness
    global best_position
    update = []
    for p in particles:
        r_own = np.random.uniform(0, 1)
        r_global = np.random.uniform(0, 1)
        prev_speed = a * p["v"]
        own_best_diff = b * r_own * -(p["pos"] - p["best_pos"])
        global_best_diff = c* r_global * -(p["pos"] - best_position)
        v = (prev_speed + own_best_diff + global_best_diff)
        v_total = np.sqrt(np.sum(np.square(v)))
        # some rather arbitrary speed clipping
        max_speed = 2
        if v_total > max_speed:
            v = (v/v_total) * max_speed
        update.append(v)
    for i, p in enumerate(particles):
        p["v"] = update[i]
        p["pos"] += update[i] * 0.01  # treat dem particles carefully
        p["fit"] = fn(p["pos"])
        if p["fit"] < p["best_fit"]:
            p["best_fit"] = deepcopy(p["fit"])
            p["best_pos"] = deepcopy(p["pos"])
        if p["fit"] < best_fitness:
            best_fitness = p["fit"]
            best_position = p["pos"]
    return particles
    

def train(fn, num_particles, num_iter, extent):
    global best_position
    global best_pos_hist
    particles = initialize(fn, num_particles, *extent)
    history = []
    for i in range(num_iter):
        a = (num_iter * 3 - i) / (num_iter * 3)
        # b = (num_iter * 5 - i) / (num_iter * 5)
        b = 2
        c = 2
        particles = update(fn, particles, a, b, c)
        history.append(deepcopy(particles))
        best_pos_hist.append(deepcopy(best_position))
        best_fit_hist.append(deepcopy(best_fitness))
    return history


def debug_pos(history):
    for state in history[0]:
        x_points = [i["pos"][0] for i in state]
        y_points = [i["pos"][1] for i in state]
        z_points = [i["fit"] for i in state]
        sns.scatterplot(x=x_points, y=y_points)
        plt.show()


def debug(history):
    global best_fitness
    global best_position
    for i, state in enumerate(history):
        v = [p["v"] for p in state]
        avg_v = np.round(np.mean(v), 3)
        avg_x = np.round(np.mean([p["pos"][0] for p in state]))
        avg_y = np.round(np.mean([p["pos"][1] for p in state]))
        if (i % 50) == 0: 
            print(f"{i} - best_position: {np.round(best_pos_hist[i], 3)} " + 
                f"({np.round(best_fit_hist[i])}) - avg speed: {avg_v}" +
                f"- avg position: ({avg_x}, {avg_y})")


if __name__ == "__main__":

    img_dir = ""
    
    extent = [-2, 2, -2, 2]
    num_particles = 20
    num_iter = 100

    # TODO: rosenbrock minimum should be 1, 1 !!!
    fn = utils.rosenbrock
    history = train(fn, num_particles, num_iter, extent)
    debug(history)
    plots.visualize_heatmap(fn, history, extent, os.path.join(img_dir, "pso_rosenbrock.gif"))

    # TODO: rastrigin minimum should be 0, 0
    fn = utils.rastrigin
    history = train(fn, num_particles, num_iter, extent)
    debug(history)
    plots.visualize_heatmap(fn, history, extent, os.path.join(img_dir, "pso_rastrigin.gif"))



