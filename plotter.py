"""Plots for the estimates and likelihoods from inference.py.

``estimates`` arguments have shape (N, num_steps+1), one row per particle, except in the MLE
validation plots at the end, where they have shape (len(N_grid), M), one row per system size N.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from scipy import stats


def plot_likelihood_parameter(test_parameters, true_value, estimated_value, likelihood, parameter):
    """Likelihood at time T (averaged over particles) against values of one parameter.

    ``likelihood`` has shape (N, len(test_parameters)), as from inference.likelihood_profile.
    """
    plt.plot(test_parameters,np.mean(likelihood,axis=0))
    plt.axvline(true_value, color='r', linestyle='dashed', linewidth=2, label=f'True {parameter}')
    plt.axvline(estimated_value, color='g', linestyle='dashed', linewidth=2, label=f'Estimated {parameter}')
    plt.xlabel(f"{parameter}")
    plt.ylabel("Likelihood value")
    plt.title(f"Likelihood function for {parameter}")
    plt.legend()
    plt.show()


def estimates_histogram(estimates, true_value, parameter):
    """Histogram of the final estimates across particles."""
    plt.hist(estimates[:,-1], bins=20, density=True, alpha=0.7)
    plt.axvline(true_value, color='r', linestyle='dashed', linewidth=2, label=f'True {parameter}')
    plt.axvline(np.mean(estimates[:,-1]), color='g', linestyle='dashed', linewidth=2, label=f'Estimated {parameter}')
    plt.title(f'Histogram of {parameter} estimates for N particles')
    plt.xlabel(f'{parameter} value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def estimates_overtime(estimates, true_value, parameter,num_steps,burn_in=1000):
    """Mean estimate across particles over time, skipping the first ``burn_in`` steps."""
    mean_estimates_over_time = np.mean(estimates, axis=0)
    plt.hlines(true_value, color='r', linestyle='dashed', linewidth=2, label=f'True {parameter}',xmin=0,xmax=num_steps)
    plt.hlines(mean_estimates_over_time[-1],color='g', linestyle='dashed', linewidth=2, label=f'Estimated {parameter}',xmin=0,xmax=num_steps)
    plt.plot(mean_estimates_over_time[burn_in:])
    plt.title(f'{parameter} Estimate vs time-steps')
    plt.xlabel('Number of time Steps')
    plt.ylabel(f'{parameter} Estimate')
    plt.show()

def L2_loss(estimates,true_value,parameter,burn_in=1000):
    """Error of the mean estimate over time, skipping the first ``burn_in`` steps."""
    mean_estimates_over_time = np.mean(estimates, axis=0)
    L2_loss_over_time = np.sqrt((mean_estimates_over_time[burn_in:] - true_value)**2)
    plt.plot(L2_loss_over_time)
    plt.title(f'L2 Loss of {parameter} Estimate Over Time')
    plt.xlabel('Number of time Steps')
    plt.ylabel('L2 Loss')
    plt.show()


def plot_y_likelihood_contour(L_surface, test_theta, test_gamma, theta_hat,gamma_hat,e_val,e_vec, true_point=None, levels=100, cmap='viridis'):
    """Contour plot of Y's likelihood in (theta_2, gamma), with the estimate and the curvature's eigenvectors.

    ``L_surface`` is from inference.likelihood_surface and ``e_val``, ``e_vec`` from inference.hessian_eig.
    ``true_point`` is the true (theta_2, gamma), marked in red if given.
    """
    th, gm = np.meshgrid(test_theta, test_gamma, indexing='ij')
    cf = plt.contourf(th, gm, L_surface, levels=levels, cmap=cmap)
    plt.plot(theta_hat,gamma_hat,'go',label='Estimate')
    if true_point is not None:
        plt.plot(*true_point,'rx',markersize=10,label='True')
    for i in range(len(e_val)):
        v_x,v_y = e_vec[:,i]
        L=max(test_gamma)
        x = np.array([theta_hat - L*v_x, theta_hat + L*v_x])
        y = np.array([gamma_hat - L*v_y, gamma_hat + L*v_y])
        plt.plot(x, y, '--', linewidth=2, label=f"λ={e_val[i]:.3g}")
    plt.xlim(test_theta[0], test_theta[-1]); plt.ylim(test_gamma[0], test_gamma[-1])
    plt.xlabel('theta_2'); plt.ylabel('gamma'); plt.colorbar(cf, label='likelihood'); plt.legend(loc='best'); plt.show()

def plot_y_likelihood_surface3d(L_surface, test_theta, test_gamma, theta_hat, gamma_hat, true_point=None, cmap='viridis'):
    """3D surface of Y's likelihood in (theta_2, gamma), with the estimate (green) and, if given, the true value (red) marked."""
    th, gm = np.meshgrid(test_theta, test_gamma, indexing='ij')
    ax = plt.figure(figsize=(8,6)).add_subplot(111, projection='3d')
    ax.plot_surface(th, gm, L_surface, cmap=cmap, edgecolor='none')
    ax.scatter(theta_hat,gamma_hat,np.max(L_surface),color='green',s=50,label='True') 
    if true_point is not None:
        ax.scatter(*true_point,np.max(L_surface),color='red',marker='x',s=50,label='True')
    ax.set_xlabel('theta_2'); ax.set_title("Likelihood profile"); ax.set_ylabel('gamma'); ax.set_zlabel('likelihood'); ax.legend(); plt.show()


def param_tex(name):
    """LaTeX symbol of a parameter name, e.g. theta_2 -> \\theta_{2}."""
    symbol, _, index = name.partition("_")
    return rf"\{symbol}_{{{index}}}" if index else rf"\{symbol}"

def _log_N_axis(ax, N_grid):
    """Log x-axis with a labelled tick at each N."""
    ax.set_xscale('log'); ax.set_xticks(N_grid); ax.set_xticklabels([str(N) for N in N_grid]); ax.set_xticks([], minor=True)
    ax.set_xlabel('Number of particles N')

def mle_validation_summary(estimates, wald_var, true_value, N_grid, var, parameter):
    """Consistency, efficiency (RMSE against the Cramer-Rao bound) and Wald coverage of the MLE against N.

    Parameters
    ----------
    estimates : ndarray, shape (len(N_grid), M)
        The MLE of each of M independent runs (simulated systems) of each size N.
    wald_var : ndarray, shape (len(N_grid), M)
        Variance of each estimate from its observed information, ((sum Q)^-1)_jj.
    true_value : float
    N_grid : array_like
    var : float
        Asymptotic variance (I_T^-1)_jj, so sqrt(var/N) is the Cramer-Rao standard deviation.
    parameter : str
        Name of the parameter, shown in the figure title.
    """
    N_grid = np.asarray(N_grid)
    M = estimates.shape[1]
    err = estimates - true_value
    crb = np.sqrt(var/N_grid)
    coverage = np.mean(np.abs(err) <= 1.96*np.sqrt(wald_var), axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    ax.fill_between(N_grid, *np.percentile(estimates, [2.5, 97.5], axis=1), alpha=0.3, label=r"Middle 95% of the runs' $\hat\theta$")
    ax.plot(N_grid, np.mean(estimates, axis=1), 'o-', label=r"Mean of the runs' $\hat\theta$")
    ax.plot(N_grid, true_value + 1.96*crb, 'k:', label=r'$\theta_0 \pm 1.96\sqrt{I_{\theta_0}^{-1}/N}$')
    ax.plot(N_grid, true_value - 1.96*crb, 'k:')
    ax.axhline(true_value, color='r', linestyle='dashed', label=r'True $\theta_0$')
    ax.set_ylabel(r'$\hat\theta$'); ax.set_title(r'Consistency: $\hat\theta \to \theta_0$')
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(N_grid, np.sqrt(np.mean(err**2, axis=1)), 'o-', label='RMSE')
    ax.plot(N_grid, np.std(err, axis=1), 's--', label='Standard deviation')
    ax.plot(N_grid, np.abs(np.mean(err, axis=1)), '^:', label='|Bias|')
    ax.plot(N_grid, crb, 'k-', label=r'Cramér-Rao bound $\sqrt{I_{\theta_0}^{-1}/N}$')
    ax.set_yscale('log'); ax.set_ylabel(r'Error of $\hat\theta$'); ax.set_title('Efficiency: RMSE against the Cramér-Rao bound')
    ax.legend(fontsize=9)

    ax = axes[2]
    band = 1.96*np.sqrt(0.95*0.05/M)
    ax.fill_between(N_grid, 0.95 - band, 0.95 + band, color='grey', alpha=0.3, label='Monte Carlo 95% band')
    ax.axhline(0.95, color='r', linestyle='dashed', label='Nominal 0.95')
    ax.plot(N_grid, coverage, 'o-', label=r'Fraction of runs covering $\theta_0$')
    ax.set_ylim(min(0.85, coverage.min() - 0.02), 1)
    ax.set_ylabel('Coverage'); ax.set_title('Asymptotic variance: Wald 95% interval coverage')
    ax.legend(fontsize=9, loc='lower right')

    for ax in axes:
        _log_N_axis(ax, N_grid)
    fig.suptitle(rf'MLE of ${param_tex(parameter)}$')
    plt.tight_layout(); plt.show()

def mle_normality(estimates, true_value, var, N_grid, parameter):
    """Asymptotic normality: for each N, a histogram of the runs' sqrt(N)(theta_hat - theta_0) against N(0, var), and a QQ plot.

    Each run contributes one MLE, estimated only from its own N particles. The top axis of each
    histogram gives theta_hat itself.

    Parameters
    ----------
    estimates : ndarray, shape (len(N_grid), M)
        The MLE of each of M independent runs of each size N.
    true_value : float
    var : float
        Asymptotic variance (I_T^-1)_jj.
    parameter : str
        As in :func:`mle_validation_summary`.
    """
    M = estimates.shape[1]
    scaled = np.sqrt(N_grid)[:, None]*(estimates - true_value)
    sd = np.sqrt(var)
    lim = max(4*sd, np.abs(scaled).max())
    h = 3.49*sd*M**(-1/3) #Scott's rule: the bin width with least mean squared error against a normal density
    k = int(np.ceil(lim/h - 0.5))
    bins = h*np.arange(-k - 0.5, k + 1) #Shared by every N, with a bin centred on 0 under the peak of the curve
    grid = np.linspace(bins[0], bins[-1], 300)
    rows = -(-len(N_grid)//2)
    fig = plt.figure(figsize=(15, 4.4*rows))
    gs = fig.add_gridspec(rows, 3)
    for i, (N, z) in enumerate(zip(N_grid, scaled)):
        ax = fig.add_subplot(gs[i//2, i % 2])
        ax.hist(z, bins=bins, density=True, alpha=0.7, label=f'{M} runs, one MLE each')
        ax.plot(grid, stats.norm.pdf(grid, 0, sd), 'r', label=r'$\mathcal{N}(0, I_{\theta_0}^{-1})$')
        ax.set_title(f'Asymptotic normality, N={N}')
        ax.set_xlabel(r'$\sqrt{N}(\hat\theta-\theta_0)$'); ax.set_ylabel('Density over runs')
        top = ax.secondary_xaxis('top', functions=(lambda z, N=N: true_value + z/np.sqrt(N), lambda t, N=N: (t - true_value)*np.sqrt(N)))
        top.set_xlabel(r'MLE $\hat\theta$ of the run', fontsize=9)
        if i == 0:
            ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[:, 2])
    q = stats.norm.ppf((np.arange(1, M+1) - 0.5)/M)
    for N, z in zip(N_grid, scaled):
        ax.plot(q, np.sort(z)/sd, '.', markersize=4, label=f'N={N}')
    ax.plot(q, q, 'k--', label='y = x')
    ax.set_xlabel(r'$\mathcal{N}(0,1)$ quantiles'); ax.set_ylabel(r'Quantiles of $\sqrt{N}(\hat\theta-\theta_0)\,/\,\sqrt{I_{\theta_0}^{-1}}$')
    ax.set_title('Asymptotic normality: normal QQ plot'); ax.legend(fontsize=9)

    fig.suptitle(rf'MLE of ${param_tex(parameter)}$')
    plt.tight_layout(); plt.show()

def mle_joint_normality(estimates, true_value, I, names, N):
    """Joint asymptotic normality of two parameters at one N: the runs' estimates against N(0, I^-1), and a chi^2 QQ plot.

    Parameters
    ----------
    estimates : ndarray, shape (M, 2)
        The joint MLE of each of M independent runs of size N.
    true_value : ndarray, shape (2,)
    I : ndarray, shape (2, 2)
        Fisher information I_T.
    names : (str, str)
    N : int
    """
    M, k = estimates.shape
    scaled = np.sqrt(N)*(estimates - true_value)
    d = np.einsum("mi,ij,mj->m", scaled, I, scaled) #Mahalanobis distances, chi^2_k asymptotically
    level = stats.chi2.ppf(0.95, k)
    symbols = [param_tex(name) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.scatter(*scaled.T, s=10, alpha=0.6, label=rf'One run each: $\sqrt{{N}}(\hat\theta-\theta_0)$ ({np.mean(d <= level):.1%} inside)')
    e_val, e_vec = np.linalg.eigh(np.linalg.inv(I))
    angle = np.linspace(0, 2*np.pi, 200)
    ellipse = e_vec @ (np.sqrt(level*e_val)[:, None]*np.stack([np.cos(angle), np.sin(angle)]))
    ax.plot(*ellipse, 'r', label=r'95% region of $\mathcal{N}(0, I_{\theta_0}^{-1})$')
    ax.set_xlabel(rf'${symbols[0]}$ component of $\sqrt{{N}}(\hat\theta-\theta_0)$')
    ax.set_ylabel(rf'${symbols[1]}$ component of $\sqrt{{N}}(\hat\theta-\theta_0)$')
    ax.set_title(f'Joint asymptotic normality: 95% region, N={N}'); ax.legend(fontsize=9)

    ax = axes[1]
    q = stats.chi2.ppf((np.arange(1, M+1) - 0.5)/M, k)
    ax.plot(q, np.sort(d), '.', markersize=4, label=f'N={N}')
    ax.plot(q, q, 'k--', label='y = x')
    ax.set_xlabel(rf'$\chi^2_{k}$ quantiles'); ax.set_ylabel(r'Quantiles of $N(\hat\theta-\theta_0)^T I_{\theta_0} (\hat\theta-\theta_0)$')
    ax.set_title(rf'Joint asymptotic normality: $\chi^2_{k}$ QQ plot'); ax.legend(fontsize=9)

    fig.suptitle(rf'Joint MLE of $({symbols[0]}, {symbols[1]})$')
    plt.tight_layout(); plt.show()
