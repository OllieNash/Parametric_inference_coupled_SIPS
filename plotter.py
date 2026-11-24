"""Plots for the estimates and likelihoods from inference.py.

``estimates`` arguments have shape (N, num_steps+1), one row per particle, except in the MLE
validation plots, where they have shape (len(N_grid), M), one row per system size N, and in the
online estimate validation plots at the end, where they have shape (len(times), M), one row per time.
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
    plt.axvline(true_value, color='r', linestyle='dashed', linewidth=2, label=rf'True ${param_tex(parameter)}$')
    plt.axvline(estimated_value, color='g', linestyle='dashed', linewidth=2, label=rf'Estimated ${param_tex(parameter)}$')
    plt.xlabel(rf"${param_tex(parameter)}$")
    plt.ylabel("Likelihood value")
    plt.title(rf"Likelihood function for ${param_tex(parameter)}$")
    plt.legend()
    plt.show()


def estimates_histogram(estimates, true_value, parameter):
    """Histogram of the final estimates across particles."""
    plt.hist(estimates[:,-1], bins=20, density=True, alpha=0.7)
    plt.axvline(true_value, color='r', linestyle='dashed', linewidth=2, label=rf'True ${param_tex(parameter)}$')
    plt.axvline(np.mean(estimates[:,-1]), color='g', linestyle='dashed', linewidth=2, label=rf'Estimated ${param_tex(parameter)}$')
    plt.title(rf'Histogram of ${param_tex(parameter)}$ estimates for N particles')
    plt.xlabel(rf'${param_tex(parameter)}$ value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def estimates_overtime(estimates, true_value, parameter,num_steps,burn_in=1000):
    """Mean estimate across particles over time, skipping the first ``burn_in`` steps."""
    mean_estimates_over_time = np.mean(estimates, axis=0)
    plt.hlines(true_value, color='r', linestyle='dashed', linewidth=2, label=rf'True ${param_tex(parameter)}$',xmin=0,xmax=num_steps)
    plt.hlines(mean_estimates_over_time[-1],color='g', linestyle='dashed', linewidth=2, label=rf'Estimated ${param_tex(parameter)}$',xmin=0,xmax=num_steps)
    plt.plot(mean_estimates_over_time[burn_in:])
    plt.title(rf'${param_tex(parameter)}$ Estimate vs time-steps')
    plt.xlabel('Number of time Steps')
    plt.ylabel(rf'${param_tex(parameter)}$ Estimate')
    plt.show()

def L2_loss(estimates,true_value,parameter,burn_in=1000):
    """Root mean squared error of the estimates across particles over time, skipping the first ``burn_in`` steps."""
    L2_loss_over_time = np.sqrt(np.mean((estimates[:, burn_in:] - true_value)**2, axis=0))
    plt.plot(L2_loss_over_time)
    plt.title(rf'L2 Loss of ${param_tex(parameter)}$ Estimate Over Time')
    plt.xlabel('Number of time Steps')
    plt.ylabel('L2 Loss')
    plt.show()


def plot_y_likelihood_contour(L_surface, test_theta, test_gamma, theta_hat,gamma_hat,e_val,e_vec, true_point=None, levels=15, cmap='viridis'):
    """Contour plot of Y's likelihood in (theta_2, gamma), with the estimate and the curvature's eigenvectors.

    ``L_surface`` is from inference.likelihood_surface and ``e_val``, ``e_vec`` from inference.hessian_eig.
    ``true_point`` is the true (theta_2, gamma), marked in red if given.
    """
    th, gm = np.meshgrid(test_theta, test_gamma, indexing='ij')
    cf = plt.contourf(th, gm, L_surface, levels=levels, cmap=cmap)
    plt.contour(th, gm, L_surface, levels=cf.levels, colors='k', linewidths=1.5, linestyles='solid') #Bold lines on the band edges
    eig_colors = ['white', 'orangered']
    for i in range(len(e_val)):
        v_x,v_y = e_vec[:,i]
        L=np.hypot(test_theta[-1] - test_theta[0], test_gamma[-1] - test_gamma[0]) #Window diagonal, so each line crosses the whole plot
        x = np.array([theta_hat - L*v_x, theta_hat + L*v_x])
        y = np.array([gamma_hat - L*v_y, gamma_hat + L*v_y])
        plt.plot(x, y, '--', color=eig_colors[i % len(eig_colors)], linewidth=2.5, label=f"λ={e_val[i]:.3g}")
    plt.plot(theta_hat,gamma_hat,'o',color='white',markeredgecolor='k',markeredgewidth=1.5,markersize=9,label='Estimate')
    if true_point is not None:
        plt.plot(*true_point,'rx',markersize=12,mew=3,label='True')
    plt.xlim(test_theta[0], test_theta[-1]); plt.ylim(test_gamma[0], test_gamma[-1])
    plt.xlabel(r'$\theta_2$'); plt.ylabel(r'$\alpha_{1,2}$'); plt.colorbar(cf, label='likelihood'); plt.legend(loc='best'); plt.show()

def plot_y_likelihood_surface3d(L_surface, test_theta, test_gamma, theta_hat, gamma_hat, true_point=None, levels=15, cmap='viridis'):
    """3D surface of Y's likelihood in (theta_2, gamma), with the estimate (green) and, if given, the true value (red) marked."""
    th, gm = np.meshgrid(test_theta, test_gamma, indexing='ij')
    ax = plt.figure(figsize=(8,6)).add_subplot(111, projection='3d')
    ax.plot_surface(th, gm, L_surface, cmap=cmap, edgecolor='none', alpha=0.85)
    ax.contour(th, gm, L_surface, levels=levels, colors='k', linewidths=1.5, linestyles='solid')
    z_min = np.min(L_surface)
    ax.contour(th, gm, L_surface, levels=levels, zdir='z', offset=z_min, cmap=cmap, linewidths=2) #Contours projected onto the floor
    ax.set_zlim(z_min, np.max(L_surface))
    ax.scatter(theta_hat,gamma_hat,np.max(L_surface),color='green',s=50,label='Estimate')
    if true_point is not None:
        ax.scatter(*true_point,np.max(L_surface),color='red',marker='x',s=50,label='True')
    ax.set_xlabel(r'$\theta_2$'); ax.set_title("Likelihood profile"); ax.set_ylabel(r'$\alpha_{1,2}$'); ax.set_zlabel('likelihood'); ax.legend(); plt.show()


#Parameters whose plotted symbol differs from their name in the code
TEX_SYMBOLS = {"gamma": r"\alpha_{1,2}"}

def param_tex(name):
    """LaTeX symbol of a parameter name, e.g. theta_2 -> \\theta_{2}, gamma -> \\alpha_{1,2}."""
    if name in TEX_SYMBOLS:
        return TEX_SYMBOLS[name]
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

def mle_joint_normality(estimates, true_value, I, names, N, title=None):
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
    title : str, optional
        Figure title, by default "Joint MLE of (...)".
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

    fig.suptitle(title or rf'Joint MLE of $({symbols[0]}, {symbols[1]})$')
    plt.tight_layout(); plt.show()


def sgd_validation_summary(estimates, lr, true_value, var, times, parameter, var_t=None, title=None):
    """Consistency of the online estimate, its error against the CLT, and the variance of (theta_hat - theta_0)/sqrt(lr) against time.

    Parameters
    ----------
    estimates : ndarray, shape (len(times), M)
        The online estimate of each of M independent runs at each time.
    lr : ndarray, shape (len(times),)
        Learning rate at each time.
    true_value : float
    var : float
        Asymptotic variance Sigma of (theta_hat - theta_0)/sqrt(lr), from inference.sgd_clt_variance,
        so sqrt(Sigma*lr) is the CLT standard deviation of theta_hat.
    times : array_like
    parameter : str
        Name of the parameter, shown in the figure title.
    var_t : ndarray, shape (len(times),), optional
        Finite-time approximation v_t of that variance, drawn dashed if given.
    title : str, optional
        Figure title, by default "Online estimate of ...".
    """
    times = np.asarray(times)
    M = estimates.shape[1]
    err = estimates - true_value
    clt_sd = np.sqrt(var*lr)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    ax.fill_between(times, *np.percentile(estimates, [2.5, 97.5], axis=1), alpha=0.3, label=r"Middle 95% of the runs' $\hat\theta_t$")
    ax.plot(times, np.mean(estimates, axis=1), '-', label=r"Mean of the runs' $\hat\theta_t$")
    ax.plot(times, true_value + 1.96*clt_sd, 'k:', label=r'$\theta_0 \pm 1.96\sqrt{\Sigma\,\mathrm{lr}_t}$')
    ax.plot(times, true_value - 1.96*clt_sd, 'k:')
    ax.axhline(true_value, color='r', linestyle='dashed', label=r'True $\theta_0$')
    ax.set_ylabel(r'$\hat\theta_t$'); ax.set_title(r'Consistency: $\hat\theta_t \to \theta_0$')
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(times, np.sqrt(np.mean(err**2, axis=1)), '-', label='RMSE')
    ax.plot(times, np.std(err, axis=1), '--', label='Standard deviation')
    ax.plot(times, np.abs(np.mean(err, axis=1)), ':', label='|Bias|')
    ax.plot(times, clt_sd, 'k-', label=r'CLT $\sqrt{\Sigma\,\mathrm{lr}_t}$')
    ax.set_yscale('log'); ax.set_ylabel(r'Error of $\hat\theta_t$'); ax.set_title('Error against the CLT standard deviation')
    ax.legend(fontsize=9)

    ax = axes[2]
    band = 1.96*var*np.sqrt(2/(M - 1))
    ax.fill_between(times, var - band, var + band, color='grey', alpha=0.3, label='Monte Carlo 95% band')
    ax.axhline(var, color='r', linestyle='dashed', label=rf'Limit $\Sigma$ = {var:.3g}')
    if var_t is not None:
        ax.plot(times, var_t, 'k--', label=r'Finite-time $v_t$')
    ax.plot(times, np.var(err/np.sqrt(lr)[:, None], axis=1), 'o-', markersize=3, label=r'Variance over the runs')
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r'Var $(\hat\theta_t-\theta_0)/\sqrt{\mathrm{lr}_t}$'); ax.set_title(r'CLT variance: $\to \Sigma$')
    ax.legend(fontsize=9)

    for ax in axes:
        ax.set_xscale('log'); ax.set_xlabel('Time t')
    fig.suptitle(title or rf'Online estimate of ${param_tex(parameter)}$')
    plt.tight_layout(); plt.show()

def sgd_error_against_N(errors, N_grid, theory_sd, T, parameter):
    """Error of the shared online estimate at time T against the number of particles N, on log-log axes.

    Parameters
    ----------
    errors : ndarray, shape (len(N_grid), M)
        theta_hat_T - theta_0 of each of M independent runs (simulated systems) of each size N.
    N_grid : array_like
    theory_sd : ndarray, shape (len(N_grid),)
        Predicted standard deviation sqrt(V_T/N), with V_T from inference.sgd_error_variance.
    T : float
    parameter : str
        Name of the parameter, shown in the title.
    """
    N_grid = np.asarray(N_grid)
    rmse = np.sqrt(np.mean(errors**2, axis=1))
    slope = np.polyfit(np.log(N_grid), np.log(rmse), 1)[0]
    _, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(N_grid, rmse, 'o-', label=f'RMSE (fitted slope {slope:.3f})')
    ax.plot(N_grid, np.std(errors, axis=1), 's--', label='Standard deviation')
    ax.plot(N_grid, np.abs(np.mean(errors, axis=1)), '^:', label='|Bias|')
    ax.plot(N_grid, theory_sd, 'k-', label=r'Theory $\sqrt{V_T/N}$ (slope $-1/2$)')
    ax.set_yscale('log'); _log_N_axis(ax, N_grid)
    ax.set_ylabel(rf'Error of $\hat{{{param_tex(parameter)}}}$')
    ax.set_title(rf'Online estimate of ${param_tex(parameter)}$ at T = {T:g}: error against N')
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()

def sgd_normality(estimates, lr, true_value, var, times, parameter):
    """CLT: for each time t, a histogram of the runs' (theta_hat_t - theta_0)/sqrt(lr_t) against N(0, var), and a QQ plot.

    The top axis of each histogram gives theta_hat_t itself.

    Parameters
    ----------
    estimates : ndarray, shape (len(times), M)
        The online estimate of each of M independent runs at each time.
    lr : ndarray, shape (len(times),)
        Learning rate at each time.
    true_value : float
    var : float
        Asymptotic variance Sigma, as in :func:`sgd_validation_summary`.
    times : array_like
    parameter : str
        As in :func:`sgd_validation_summary`.
    """
    M = estimates.shape[1]
    scaled = (estimates - true_value)/np.sqrt(lr)[:, None]
    sd = np.sqrt(var)
    lim = max(4*sd, np.abs(scaled).max())
    h = 3.49*sd*M**(-1/3) #Scott's rule, as in mle_normality
    k = int(np.ceil(lim/h - 0.5))
    bins = h*np.arange(-k - 0.5, k + 1) #Shared by every t, with a bin centred on 0
    grid = np.linspace(bins[0], bins[-1], 300)
    rows = -(-len(times)//2)
    fig = plt.figure(figsize=(15, 4.4*rows))
    gs = fig.add_gridspec(rows, 3)
    for i, (t, l, z) in enumerate(zip(times, lr, scaled)):
        ax = fig.add_subplot(gs[i//2, i % 2])
        ax.hist(z, bins=bins, density=True, alpha=0.7, label=f'{M} runs, one estimate each')
        ax.plot(grid, stats.norm.pdf(grid, 0, sd), 'r', label=r'$\mathcal{N}(0, \Sigma)$')
        ax.set_title(f'CLT for the online estimate, t={t:g}')
        ax.set_xlabel(r'$(\hat\theta_t-\theta_0)/\sqrt{\mathrm{lr}_t}$'); ax.set_ylabel('Density over runs')
        top = ax.secondary_xaxis('top', functions=(lambda z, l=l: true_value + z*np.sqrt(l), lambda e, l=l: (e - true_value)/np.sqrt(l)))
        top.set_xlabel(r'Online estimate $\hat\theta_t$ of the run', fontsize=9)
        if i == 0:
            ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[:, 2])
    q = stats.norm.ppf((np.arange(1, M+1) - 0.5)/M)
    for t, z in zip(times, scaled):
        ax.plot(q, np.sort(z)/sd, '.', markersize=4, label=f't={t:g}')
    ax.plot(q, q, 'k--', label='y = x')
    ax.set_xlabel(r'$\mathcal{N}(0,1)$ quantiles'); ax.set_ylabel(r'Quantiles of $(\hat\theta_t-\theta_0)\,/\sqrt{\Sigma\,\mathrm{lr}_t}$')
    ax.set_title('CLT: normal QQ plot'); ax.legend(fontsize=9)

    fig.suptitle(rf'Online estimate of ${param_tex(parameter)}$')
    plt.tight_layout(); plt.show()
