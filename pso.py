"""
initialize swarm, update location on each iteration
"""
import numpy as np
from copy import deepcopy
from plots import rastrigin

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
        p["fit"] = fitness_func(p["pos"])
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
    history = train(2, 10, rastrigin)