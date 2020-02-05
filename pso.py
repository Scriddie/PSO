"""
initialize swarm, update location on each iteration
"""
import numpy as np

def initialize(n, x_min, x_max, y_min, y_max):
    initial_positions = [
        (np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max))
        for i in range(n)
    ]
    return initial_positions

def update(positions):
    pass

if __name__ == "__main__":
    initialize(10, -2, 2, -2, 2)