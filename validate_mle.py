"""Monte Carlo validation of the MLE as the number of particles N grows, at fixed T.

For each N there are M independent runs. Each run simulates its own interacting particle system
(IPS) of N particles on [0, T] and gives one MLE theta_hat. The confining terms use the centres
c_x, c_y from the parameters, and the interaction terms use the run's empirical means m_x, m_y at each
time step. The M estimates per N are then used to check consistency, the RMSE against the
Cramer-Rao bound, the coverage of Wald intervals, and asymptotic normality
sqrt(N)(theta_hat - theta_0) -> N(0, I_T^-1), where I_T is the Fisher information of one
particle's path on [0, T] in the N -> infinity (McKean-Vlasov) limit.

Usage: python validate_mle.py [ou | mv_ou | coupled]   (default: coupled)
"""
import sys

import numpy as np

from inference import fisher_information, likelihood_coefficients, mle_per_run
from model import PARAMS, PRESETS, empirical_means, simulate
from plotter import mle_joint_normality, mle_normality, mle_validation_summary

PRESET = sys.argv[1] if len(sys.argv) > 1 else "coupled"

T = 50 #Time horizon of every system
N_GRID = [10, 50, 200, 1000] #System sizes
M = 200 #Independent runs per N; each run simulates its own N particles and gives one MLE
CHUNK = 5000 #Most particles per call to simulate, to bound memory

#Not model.setup, which also simulates a separate IPS for fixed means; every run here uses its own empirical means
params = {**PARAMS, **PRESETS[PRESET]["params"], "T": T}
targets = PRESETS[PRESET]["estimate"]

#(population, parameters estimated jointly)
groups = [(pop, (param,)) for pop, param in targets]

#MLE of each run (len(N_GRID), M, k) and its observed information (len(N_GRID), M, k, k) for each group.
#Runs are simulated in batches; each run is a separate system in simulate, with its own particles.
estimates = {g: [] for g in groups}
info = {g: [] for g in groups}
for N in N_GRID:
    est_N, info_N = {g: [] for g in groups}, {g: [] for g in groups}
    for start in range(0, M, max(1, CHUNK//N)):
        runs = min(M - start, max(1, CHUNK//N))
        paths = simulate({**params, "N": N}, None, systems=runs)
        means = empirical_means(paths, N)
        for pop, names in groups:
            _, l, Q = likelihood_coefficients(paths, pop, list(names), params, means)
            e, i = mle_per_run(l, Q, N)
            est_N[pop, names].append(e)
            info_N[pop, names].append(i)
    for g in groups:
        estimates[g].append(np.concatenate(est_N[g]))
        info[g].append(np.concatenate(info_N[g]))

print(f"\nPreset {PRESET}, T={T}, {M} runs per N (one MLE per run), empirical means from the IPS")
print(f"c_x = {params['c_x']}, c_y = {params['c_y']}")

results = []
for pop, names in groups:
    est, obs = np.array(estimates[pop, names]), np.array(info[pop, names])
    true = np.array([params[p] for p in names])
    I_T = fisher_information(pop, names, params)
    #I_T is singular when the parameters are not separately identifiable, e.g. theta_2 and gamma from Y with
    #c_x = c_y. Their asymptotic variances are then infinite, so there is no Cramer-Rao bound or normal limit
    identifiable = np.linalg.matrix_rank(I_T) == len(names)
    I_inv = np.linalg.inv(I_T) if identifiable else np.full_like(I_T, np.inf)
    wald_var = np.diagonal(np.linalg.inv(obs), axis1=-2, axis2=-1)
    results.append((pop, names, est, true, I_T, I_inv, wald_var, identifiable))

    print(f"\n{', '.join(names)} from {pop}")
    print(f"Fisher information I_T:\n{I_T}")
    if not identifiable:
        null = np.linalg.eigh(I_T)[1][:, 0]
        print(f"I_T is singular: {', '.join(names)} are not separately identifiable from {pop}, the likelihood is flat "
              f"along {null.round(4)}. No Cramer-Rao bound or plots for them")
    print(f"Mean observed information per particle at N={N_GRID[-1]}:\n{obs[-1].mean(axis=0)/N_GRID[-1]}")
    for j, name in enumerate(names):
        print(f"\n{name} (true {true[j]})")
        print(f"{'N':>6} {'mean':>9} {'bias':>9} {'sd':>8} {'RMSE':>8} {'CR sd':>8} {'coverage':>8}")
        for n, N in enumerate(N_GRID):
            err = est[n, :, j] - true[j]
            coverage = np.mean(np.abs(err) <= 1.96*np.sqrt(wald_var[n, :, j]))
            print(f"{N:>6} {np.mean(est[n, :, j]):>9.4f} {np.mean(err):>+9.4f} {np.std(err):>8.4f} "
                  f"{np.sqrt(np.mean(err**2)):>8.4f} {np.sqrt(I_inv[j, j]/N):>8.4f} {coverage:>8.3f}")

for pop, names, est, true, I_T, I_inv, wald_var, identifiable in results:
    if not identifiable:
        continue
    for j, name in enumerate(names):
        mle_validation_summary(est[:, :, j], wald_var[:, :, j], true[j], N_GRID, I_inv[j, j], name)
        mle_normality(est[:, :, j], true[j], I_inv[j, j], N_GRID, name)
    if len(names) == 2:
        mle_joint_normality(est[-1], true, I_T, names, N_GRID[-1])
