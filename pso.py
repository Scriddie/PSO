"""
initialize swarm, update location on each iteration
"""
import numpy as np
from copy import deepcopy
import plots
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import imageio
from importlib import reload
reload(plots)
reload(utils)

best_fitness = np.inf
best_fit_hist = [best_fitness]
best_position = np.array([0, 0])
best_pos_hist = [best_position]

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
    # TODO: this doesbt seem to work
    for p in particles:
        p["best_pos"] = (p["pos"])
        p["fit"] = fn(p["pos"][0], p["pos"][1])
        p["best_fit"] = p["fit"]
        if p["fit"] < best_fitness:
            best_fitness = p["fit"]
            best_position = p["pos"]
    return particles


def update(fn, particles):
    global best_fitness
    global best_position
    update = []
    for p in particles:
        r_own = np.random.uniform(0, 1)
        r_global = np.random.uniform(0, 1)
        prev_speed = p["v"]
        own_best_diff = r_own * -(p["pos"] - p["best_pos"])
        global_best_diff = r_global * -(p["pos"] - best_position)
        v = (prev_speed + own_best_diff + global_best_diff)
        v_total = np.sqrt(np.sum(np.square(v)))
        # some rather arbitrary speed clipping
        max_speed = 2
        if v_total > max_speed:
            v = (v/v_total) * max_speed
        update.append(v)
    # TODO: seems like speed is changing but direction is not?
    for i, p in enumerate(particles):
        p["v"] = update[i]
        p["pos"] += update[i] * 0.01
        p["fit"] = fn(p["pos"][0], p["pos"][1])
        if p["fit"] < p["best_fit"]:
            p["best_fit"] = p["fit"]
            p["best_pos"] = p["pos"]
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
        particles = update(fn, particles)
        history.append(deepcopy(particles))
        best_pos_hist.append(deepcopy(best_position))
        best_fit_hist.append(deepcopy(best_fitness))
    return history

def debug(history):
    global best_fitness
    global best_position
    for i, state in enumerate(history):
        v = [p["v"] for p in state]
        avg_v = np.round(np.mean(v), 3)
        avg_x = np.round(np.mean([p["pos"][0] for p in state]))
        avg_y = np.round(np.mean([p["pos"][1] for p in state]))
        print(f"{i}\t best_position: {np.round(best_pos_hist[i], 3)} " + 
              f"({np.round(best_fit_hist[i])}) - avg speed: {avg_v}" +
              f"- avg position: ({avg_x}, {avg_y})")


if __name__ == "__main__":
    fn = utils.rosenbrock
    extent = [-2, 2, -2, 2]
    history = train(fn, 10, 200, extent)
    debug(history)
    # plots.visualize_3D(utils.rastrigin, history)
    plots.visualize_heatmap(fn, history, extent)



