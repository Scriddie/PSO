from scipy.optimize import minimize
import utils

utils.rastrigin(0, 0)

a = minimize(utils.rastrigin, 1, 2)