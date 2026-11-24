"""Monte Carlo validation of the online (stochastic gradient) estimates, as time grows or as N grows.

Time study (default). There are M independent runs. Each run simulates one particle of X and of Y with
the interaction means held fixed, as in run_sgd.py, and updates its own online estimate at every time step
with learning rate lr_t = lr0*max(t, 1)^-d. Only the current positions and estimates are kept, so memory
does not grow with T. The M estimates at each recorded time are used to check consistency,
theta_hat_t -> theta_0, and the central limit theorem

    (theta_hat_t - theta_0)/sqrt(lr_t) -> N(0, Sigma),

where Sigma = 1/2 for d in (0, 1), the same for every parameter, and
Sigma = 1/(2 - 1/(lr0 (beta/2) E[phi^2])) for d = 1 (see inference.sgd_clt_variance).

N study. For each N in a grid there are M_N independent runs. Each run simulates its own interacting
particle system (IPS) of N particles on [0, T_N], with the run's empirical means in the interactions as in
validate_mle.py, and keeps one online estimate, updated at every step with the gradient averaged over its
N particles (inference.sgd_step_shared). Averaging divides the noise of the error recursion by N, so at
fixed T_N the RMSE should fall like N^-1/2, towards sqrt(V_T/N) (inference.sgd_error_variance).

Gain study (natural or sgd). As the time study, but each run estimates (theta_2, gamma) from Y, either jointly or
each alone with the other known, with p <- p + lr G^-1 grad (inference.natural_step). For "natural", G is the
running empirical Fisher information or the exact one i = (beta/2) E[phi phi^T]; for "sgd", G = I, plain gradient
ascent. Then (theta_hat_t - theta_0)/sqrt(lr_t) -> N(0, Sigma), with Sigma from a Lyapunov equation
(inference.gain_clt_covariance). For the natural gradient with lr_t = 1/t, Sigma = i^-1, so Var(theta_hat_t) = i^-1/t,
the Cramer-Rao bound: (i^-1)_jj/t jointly, 1/(i_jj t) alone. The correlation of phi(X_u) with past noise gives the
exact gain a bias of about 3 log(t)/t, while the empirical gain, like the MLE, has an O(1/t) bias.

Usage: python validate_sgd.py [time | N | natural | sgd] [joint | individual]   (default: time, joint)
"""
import sys

import numpy as np
from tqdm import trange

from inference import (
    fisher_information,
    gain_clt_covariance,
    learning_rate,
    natural_step,
    sgd_clt_variance,
    sgd_error_variance,
    sgd_step,
    sgd_step_shared,
)
from model import PARAMS, POPULATIONS, _system_means, em_step, ips_means
from plotter import mle_joint_normality, param_tex, sgd_error_against_N, sgd_normality, sgd_validation_summary

MODE = sys.argv[1] if len(sys.argv) > 1 else "time"
JOINT = (sys.argv[2] if len(sys.argv) > 2 else "joint") == "joint" #Gain study: joint or individual estimation

params = {**PARAMS, "T": 5000, "c_x": 2} #As in run_sgd.py
M = 2000 #Time study: independent runs; each is one particle with its own online estimate
CLT_TIMES = [10, 100, 1000, 5000] #Time study: times of the CLT histograms
T_N = 50 #N study: time horizon of every system, as in validate_mle.py
N_GRID = [10, 20, 50, 100, 200, 500, 1000] #N study: system sizes
M_N = 200 #N study: independent runs per N; each simulates its own N particles and keeps one online estimate

#(population, parameter, initial learning rate, learning rate decay in time), as in run_sgd.py but with decay 0.75:
#the variance of (theta_hat - theta_0)/sqrt(lr) approaches its limit like t^-(1 - decay), so with 0.9 it is still far off at T
TARGETS = [
    ("y", "theta_2", 2.5, 0.75),
    ("x", "gamma", 5, 0.75),
]
#Gain study: (population, parameters, initial learning rate, learning rate decay) of each update
GAIN_STUDY = {
    "natural": ("y", ["theta_2", "gamma"], 1, 1), #lr_t = 1/t makes the estimate efficient (Cramer-Rao)
    "sgd": ("y", ["theta_2", "gamma"], 2.5, 0.75), #Plain gradient, as TARGETS
}

h = params["h"]


def validate_in_time():
    """Consistency and CLT of the online estimate as t grows; see the module docstring."""
    num_steps = int(params["T"]/h)
    #Steps after which the estimates are recorded: log-spaced for the consistency plots, plus CLT_TIMES
    record = np.unique(np.round(np.concatenate([np.geomspace(1, params["T"], 50), CLT_TIMES])/h).astype(int))
    times = record*h
    clt = np.searchsorted(record, np.round(np.array(CLT_TIMES)/h).astype(int))

    means = ips_means(params, N=5000, T=100) #Fixed means for the interactions; they relax at rate >= 1, so T=100 is plenty

    #With fixed means the populations are independent, so only those with a target are simulated
    pops = sorted({pop for pop, *_ in TARGETS})
    X = {pop: np.random.normal(0, params["ic_scale"], size=M) for pop in pops} #Initial distribution, as in model.simulate
    #Initial estimates, uniform on the true value +- 2 as in run_sgd.py
    estimates = {target: params[target[1]] + np.random.uniform(-2, 2, size=M) for target in TARGETS}
    recorded = {target: np.zeros((len(record), M)) for target in TARGETS}
    r = 0
    for k in trange(num_steps, desc="Simulating and estimating online"):
        X_next = {pop: em_step(pop, X[pop], params, means) for pop in pops}
        for target in TARGETS:
            pop, param, lr0, lr_decay = target
            estimates[target] = sgd_step(estimates[target], X[pop], X_next[pop], learning_rate(k*h, lr0, lr_decay), pop, param, params, means)
        X = X_next
        if r < len(record) and k+1 == record[r]:
            for target in TARGETS:
                recorded[target][r] = estimates[target]
            r += 1

    print(f"\n{M} runs of one particle each, T={params['T']}, fixed means {means}")

    results = []
    for target in TARGETS:
        pop, param, lr0, lr_decay = target
        est, true = recorded[target], params[param]
        lr = learning_rate(times, lr0, lr_decay)
        #E[phi^2] averaged over [0, 100], which is essentially its stationary value
        phi2 = fisher_information(pop, [param], {**params, "T": 100}, means)[0, 0]/(params["beta"]/2*100)
        Sigma = float(sgd_clt_variance(np.inf, lr0, lr_decay, phi2, params["beta"]))
        #v_t is a large-t approximation; before t = 10 the spread of the initial estimates dominates
        v_t = np.where(times >= 10, sgd_clt_variance(times, lr0, lr_decay, phi2, params["beta"]), np.nan)
        z = (est - true)/np.sqrt(lr)[:, None]
        results.append((param, est, true, lr, Sigma, v_t))

        print(f"\n{param} from {pop} (true {true}), lr_t = {lr0}*max(t, 1)^-{lr_decay}, E[phi^2] = {phi2:.4f}")
        if not np.isfinite(Sigma):
            print(f"beta*lr0*E[phi^2] <= 1: the error decays slower than sqrt(lr_t), so there is no CLT at this rate. "
                  f"Increase lr0 or use a decay below 1")
            continue
        print(f"Limit variance of (theta_hat - theta_0)/sqrt(lr): Sigma = {Sigma:.4f}")
        print(f"{'t':>6} {'mean':>8} {'bias':>9} {'sd':>8} {'RMSE':>8} {'CLT sd':>8} {'Var z':>7} {'v_t':>7} {'coverage':>8}")
        #At finite t, Var z is v_t > Sigma and there is an O(lr_t) bias from the correlation of phi(X_t) with
        #past noise. Both vanish relative to sqrt(lr_t)
        for i in clt:
            err = est[i] - true
            coverage = np.mean(np.abs(z[i]) <= 1.96*np.sqrt(Sigma))
            print(f"{times[i]:>6g} {np.mean(est[i]):>8.4f} {np.mean(err):>+9.4f} {np.std(err):>8.4f} {np.sqrt(np.mean(err**2)):>8.4f} "
                  f"{np.sqrt(Sigma*lr[i]):>8.4f} {np.var(z[i]):>7.4f} {v_t[i]:>7.4f} {coverage:>8.3f}")
        late = times >= 1000
        slope = np.polyfit(np.log(times[late]), np.log(np.sqrt(np.mean((est[late] - true)**2, axis=1))), 1)[0]
        print(f"RMSE ~ t^{slope:.3f} for t >= 1000 (CLT: -decay/2 = {-lr_decay/2})")

    for param, est, true, lr, Sigma, v_t in results:
        if not np.isfinite(Sigma):
            continue
        sgd_validation_summary(est, lr, true, Sigma, times, param, v_t)
        sgd_normality(est[clt], lr[clt], true, Sigma, times[clt], param)


def validate_in_N():
    """Error of the shared online estimate at time T_N against the number of particles N; see the module docstring."""
    p = {**params, "T": T_N}
    errors = {target: np.zeros((len(N_GRID), M_N)) for target in TARGETS}
    for n, N in enumerate(N_GRID):
        #Rows s*N to (s+1)*N - 1 are run s, as in model.simulate. Both populations are simulated, since the
        #empirical means couple them
        X = {pop: np.random.normal(0, p["ic_scale"], size=M_N*N) for pop in POPULATIONS}
        #Start at the true value: the transient from a random start decays like exp(-i int lr) whatever N, and for
        #gamma it is still a few percent at T_N, which would put a floor under the error at large N
        estimates = {target: np.full(M_N, float(p[target[1]])) for target in TARGETS}
        for k in trange(int(T_N/h), desc=f"N = {N}"):
            means = {pop: _system_means(x, N) for pop, x in X.items()}
            X_next = {pop: em_step(pop, X[pop], p, means) for pop in POPULATIONS}
            for target in TARGETS:
                pop, param, lr0, lr_decay = target
                estimates[target] = sgd_step_shared(estimates[target], X[pop], X_next[pop], learning_rate(k*h, lr0, lr_decay), N, pop, param, p, means)
            X = X_next
        for target in TARGETS:
            errors[target][n] = estimates[target] - p[target[1]]

    print(f"\n{M_N} runs per N, each an IPS of N particles with one online estimate, T={T_N}, "
          f"c_x = {p['c_x']}, c_y = {p['c_y']}, empirical means")

    results = []
    for target in TARGETS:
        pop, param, lr0, lr_decay = target
        err = errors[target]
        #E[phi^2] in the McKean-Vlasov limit, which the runs' empirical means approach, averaged over [0, 100]
        phi2 = fisher_information(pop, [param], {**p, "T": 100})[0, 0]/(p["beta"]/2*100)
        theory_sd = np.sqrt(sgd_error_variance(T_N, h, lr0, lr_decay, phi2, p["beta"])/np.array(N_GRID))
        results.append((param, err, theory_sd))

        print(f"\n{param} from {pop} (true {p[param]}), lr_t = {lr0}*max(t, 1)^-{lr_decay}, E[phi^2] = {phi2:.4f}")
        print(f"{'N':>6} {'mean':>9} {'bias':>9} {'sd':>8} {'RMSE':>8} {'theory':>8}")
        for n, N in enumerate(N_GRID):
            print(f"{N:>6} {p[param] + np.mean(err[n]):>9.4f} {np.mean(err[n]):>+9.4f} {np.std(err[n]):>8.4f} "
                  f"{np.sqrt(np.mean(err[n]**2)):>8.4f} {theory_sd[n]:>8.4f}")
        slope = np.polyfit(np.log(N_GRID), np.log(np.sqrt(np.mean(err**2, axis=1))), 1)[0]
        print(f"RMSE ~ N^{slope:.3f} (theory: -1/2)")

    for param, err, theory_sd in results:
        sgd_error_against_N(err, N_GRID, theory_sd, T_N, param)


def validate_gains(update, joint):
    """Consistency, CLT and efficiency of the gain study as t grows; see the module docstring.

    ``update`` is a key of GAIN_STUDY, and ``joint`` says whether the parameters are estimated jointly or each alone.
    """
    pop, names, lr0, lr_decay = GAIN_STUDY[update]
    true = np.array([params[p] for p in names], dtype=float)
    num_steps = int(params["T"]/h)
    clt_times = [t for t in CLT_TIMES if t <= params["T"]]
    record = np.unique(np.round(np.concatenate([np.geomspace(1, params["T"], 50), clt_times])/h).astype(int))
    times = record*h
    clt = np.searchsorted(record, np.round(np.array(clt_times)/h).astype(int))

    means = ips_means(params, N=5000, T=100) #As in validate_in_time
    #Fisher information per unit time, averaged over [0, 100], which is essentially its stationary value
    fisher = fisher_information(pop, names, {**params, "T": 100}, means)/100
    gains = {"empirical": None, "exact": fisher} if update == "natural" else {"plain": np.eye(len(names))}
    #Indices of the parameters estimated together: all at once, or each alone with the others known
    blocks = [list(range(len(names)))] if joint else [[j] for j in range(len(names))]
    runs = [(g, tuple(idx)) for g in gains for idx in blocks]
    gain = {(g, idx): None if gains[g] is None else gains[g][np.ix_(idx, idx)] for g, idx in runs}

    X = np.random.normal(0, params["ic_scale"], size=M)
    initial = true + np.random.uniform(-2, 2, size=(M, len(names))) #As in validate_in_time, the same start for every run
    estimates = {run: initial[:, list(run[1])] for run in runs}
    info = {run: np.tile(np.eye(len(run[1])), (M, 1, 1)) for run in runs} #Running observed information, starting at the identity
    recorded = {run: np.zeros((len(record), M, len(run[1]))) for run in runs}
    r = 0
    for step in trange(num_steps, desc=f"Simulating and estimating online ({update}, {'joint' if joint else 'individual'})"):
        X_next = em_step(pop, X, params, means)
        lr = learning_rate(step*h, lr0, lr_decay)
        for run in runs:
            estimates[run], info[run] = natural_step(estimates[run], info[run], step*h, X, X_next, lr, pop,
                                                     [names[j] for j in run[1]], params, means, gain[run])
        X = X_next
        if r < len(record) and step+1 == record[r]:
            for run in runs:
                recorded[run][r] = estimates[run]
            r += 1

    lr = learning_rate(times, lr0, lr_decay)
    label = {"natural": "Natural gradient", "sgd": "Plain gradient"}[update]
    print(f"\n{M} runs of one particle each, T={params['T']}, fixed means {means}")
    print(f"{label} for {names} from {pop}, {'jointly' if joint else 'each alone'}, lr_t = {lr0}*max(t, 1)^-{lr_decay}, "
          f"Fisher information per unit time\n{fisher}")

    for g, idx in runs:
        block = [names[j] for j in idx]
        F = fisher[np.ix_(idx, idx)]
        Sigma = gain_clt_covariance(np.inf, lr0, lr_decay, F, gain[g, idx])
        run_label = f"{', '.join(block)}" + (f", {g} gain" if update == "natural" else "")
        if not np.all(np.isfinite(Sigma)):
            print(f"\n{run_label}: the error decays slower than sqrt(lr_t), so there is no CLT at this rate. Increase lr0 or use a decay below 1")
            continue
        #Finite-time covariance; before t = 10 the spread of the initial estimates dominates
        Sigma_t = np.array([gain_clt_covariance(t, lr0, lr_decay, F, gain[g, idx]) if t >= 10 else np.full_like(F, np.nan) for t in times])
        cr = np.diag(np.linalg.inv(F)) #Cramer-Rao variance per unit time of each parameter, given the others in the block
        print(f"\n{run_label}: limit covariance of (theta_hat - theta_0)/sqrt(lr), Sigma =\n{Sigma}")
        for j, name in enumerate(block):
            est, true_j = recorded[g, idx][:, :, j], true[idx[j]]
            z = (est - true_j)/np.sqrt(lr)[:, None]
            print(f"\n{name} (true {true_j})")
            print(f"{'t':>6} {'mean':>8} {'bias':>9} {'sd':>8} {'RMSE':>8} {'CLT sd':>8} {'CR sd':>8} {'coverage':>8}")
            for i in clt:
                err = est[i] - true_j
                coverage = np.mean(np.abs(z[i]) <= 1.96*np.sqrt(Sigma[j, j]))
                print(f"{times[i]:>6g} {np.mean(est[i]):>8.4f} {np.mean(err):>+9.4f} {np.std(err):>8.4f} {np.sqrt(np.mean(err**2)):>8.4f} "
                      f"{np.sqrt(Sigma[j, j]*lr[i]):>8.4f} {np.sqrt(cr[j]/times[i]):>8.4f} {coverage:>8.3f}")
            title = rf"{label} estimate of ${param_tex(name)}$ ({'joint' if joint else 'alone'})" + (f", {g} gain" if update == "natural" else "")
            sgd_validation_summary(est, lr, true_j, Sigma[j, j], times, name, Sigma_t[:, j, j], title=title)
        if len(idx) > 1:
            symbols = ", ".join(param_tex(p) for p in block)
            #The scaled error is theta_hat - theta_0 (N = 1), against N(0, Sigma lr_T)
            mle_joint_normality(recorded[g, idx][-1], true[list(idx)], np.linalg.inv(Sigma*lr[-1]), block, 1,
                                title=rf"{label} estimate of $({symbols})$" + (f", {g} gain" if update == "natural" else "") + f", t = {times[-1]:g}")


if MODE in GAIN_STUDY:
    validate_gains(MODE, JOINT)
else:
    {"time": validate_in_time, "N": validate_in_N}[MODE]()
