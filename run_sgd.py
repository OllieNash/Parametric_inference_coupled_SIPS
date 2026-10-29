"""Online (stochastic gradient) estimation of theta_2 and gamma for the coupled model.

Usage: python run_sgd.py
"""
import numpy as np

from inference import likelihood_profile, sgd_path
from model import PARAMS, ips_means, simulate
from plotter import L2_loss, estimates_overtime, plot_likelihood_parameter

params = {**PARAMS, "T": 3000, "mu_x": 0.5}
num_steps = int(params["T"]/params["h"])
means = ips_means(params, N=5000, T=30)
paths = simulate(params, means)

#(population, parameter, learning rate decay, values to plot the likelihood at final time against)
targets = [
    ("y", "theta_2", 0.125, np.linspace(-10,10,5000)),
    ("x", "gamma", 0.085, np.linspace(-5,5,5000)),
]

for pop, param, lr_decay, test_values in targets:
    initial = np.random.uniform(-0.5,4.5,size=params["N"]) #Initial distribution for the estimate
    estimates = sgd_path(paths, pop, param, params, means, initial, lr_decay)
    print(f"{param} estimate {estimates[:,-1]} (true {params[param]})")

    likelihood = likelihood_profile(paths, pop, param, test_values, params, means)
    plot_likelihood_parameter(test_values, params[param], np.mean(estimates[:,-1]), likelihood, param)
    estimates_overtime(estimates, params[param], param, num_steps)
    L2_loss(estimates, params[param], param)

print(f"Means used for the interactions: {means}")
