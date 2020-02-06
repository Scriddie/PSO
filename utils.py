import numpy as np

def rosenbrock(x1, x2, a=0, b=1000):
    return (a - x1) ** 2 + b * (x2 - x1 ** 2) ** 2

def rastrigin(x1, x2, a=1):
    X = np.stack((x1, x2), axis=-1)
    
    res = X ** 2 - a * np.cos(2 * np.pi * X)
    return 2 * a + np.sum(res, axis=-1)