"""Offline maximum likelihood estimation for one of the presets in model.PRESETS.

Usage: python run_mle.py [ou | mv_ou | coupled]   (default: coupled)
"""
import sys

import numpy as np

from inference import hessian_eig, likelihood_coefficients, likelihood_profile, likelihood_surface, mle_path
from model import setup, simulate
from plotter import (
    L2_loss,
    estimates_histogram,
    estimates_overtime,
    plot_likelihood_parameter,
    plot_y_likelihood_contour,
    plot_y_likelihood_surface3d,
)

PRESET = sys.argv[1] if len(sys.argv) > 1 else "coupled"

params, targets, means = setup(PRESET)
num_steps = int(params["T"]/params["h"])
paths = simulate(params, means)

test_values = np.linspace(-20,20,1000) #We plot the likelihood at final time against these
burn_in = 1000 if PRESET == "coupled" else 50

estimates = {}
for pop, param in targets:
    estimates[param] = mle_path(paths, pop, param, params, means)
    estimate = np.mean(estimates[param][:,-1])
    print(f"Estimated {param}: {estimate} (true {params[param]})")
    print(f"{param} standard deviation: {np.std(estimates[param][:,-1])}")

    likelihood = likelihood_profile(paths, pop, param, test_values, params, means)
    plot_likelihood_parameter(test_values, params[param], estimate, likelihood, param)
    if params["N"] > 1:
        estimates_histogram(estimates[param], params[param], param)
    estimates_overtime(estimates[param], params[param], param, num_steps, burn_in)
    L2_loss(estimates[param], params[param], param, burn_in)

print(f"Means used for the interactions: {means}")

#Likelihood of Y against the confining and cross-interaction parameters, with the curvature's eigenvectors
if PRESET == "coupled":
    names = ("theta_2", "gamma")
    e_val, e_vec = hessian_eig(paths, "y", names, params, means)
    print([(e_val[i], e_vec[:, i]) for i in range(len(e_val))])

    #Joint MLE of (theta_2, gamma) from Y alone, i.e. the maximum of the surface
    _, l, Q = likelihood_coefficients(paths, "y", names, params, means)
    theta_hat, gamma_hat = np.linalg.solve(Q.sum(axis=0), l.sum(axis=0))
    print(f"Joint MLE from Y: theta_2 {theta_hat}, gamma {gamma_hat} (true {params['theta_2']}, {params['gamma']})")

    surface_values = np.linspace(-5,5,500)
    true_point = (params["theta_2"], params["gamma"])
    surface = likelihood_surface(paths, "y", names, (surface_values, surface_values), params, means)
    plot_y_likelihood_surface3d(surface, surface_values, surface_values, theta_hat, gamma_hat, true_point)
    plot_y_likelihood_contour(surface, surface_values, surface_values, theta_hat, gamma_hat, e_val, e_vec, true_point)
