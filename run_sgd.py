"""Online (stochastic gradient) estimation of theta_2 and gamma for the coupled model.

Usage: python run_sgd.py
"""
import numpy as np

from inference import likelihood_profile, sgd_path
from model import PARAMS, ips_means, simulate
from plotter import L2_loss, estimates_overtime, plot_likelihood_parameter

params = {**PARAMS, "T": 5000, "c_x": 2}
num_steps = int(params["T"]/params["h"])
means = ips_means(params, N=5000, T=1000)
paths = simulate(params, means)

#(population, parameter, initial learning rate, learning rate decay in time, values to plot the likelihood at final time against)
#Any decay in (0, 1] converges, with error of order sqrt(lr), so a small decay converges slowly;
#gamma's basis function is smaller, so it needs a larger initial rate
targets = [
    ("y", "theta_2", 2.5, 0.9, np.linspace(-10,10,5000)),
    ("x", "gamma", 5, 0.9, np.linspace(-5,5,5000)),
]

for pop, param, lr0, lr_decay, test_values in targets:
    initial = params[param] + np.random.uniform(-2,2,size=params["N"]) #Initial distribution for the estimate, centred at the true value
    estimates = sgd_path(paths, pop, param, params, means, initial, lr_decay, lr0)
    print(f"{param} estimate {estimates[:,-1]} (true {params[param]})")

    likelihood = likelihood_profile(paths, pop, param, test_values, params, means)
    plot_likelihood_parameter(test_values, params[param], np.mean(estimates[:,-1]), likelihood, param)
    estimates_overtime(estimates, params[param], param, num_steps)
    L2_loss(estimates, params[param], param)

print(f"Means used for the interactions: {means}")
