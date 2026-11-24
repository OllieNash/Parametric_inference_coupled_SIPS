"""Offline MLE, online gradient estimation and the likelihood for the model in model.py.

The drift is linear in every parameter p, b = b_rest + sum_p p*phi_p, so both methods only need
phi_p(X_t) and the increments dX_t. All sums are left-point (Ito) sums.

Common arguments: ``paths`` is the output of :func:`model.simulate`, ``pop`` is "x" or "y",
and ``params``, ``means`` are as in model.py. The values of ``means`` may also be arrays that
broadcast against X_t, e.g. the empirical means from :func:`model.empirical_means`.
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from tqdm import trange

from model import POPULATIONS, basis, diffusion, drift


def _increments(paths, pop, names, params, means):
    """X_t, dX_t (N, T), phi_p(X_t) for p in names (k, N, T), and the drift without those parameters (N, T)."""
    X = paths[pop]
    X_t, dX = X[:, :-1], np.diff(X, axis=1)
    phi = basis(pop, X_t, params, means)
    b_rest = drift(pop, X_t, params, means) - sum(params[p]*phi[p] for p in names)
    return X_t, dX, np.stack([phi[p] for p in names]), b_rest


def mle_path(paths, pop, param, params, means):
    """Offline MLE of one parameter, the others known, using the path up to each time step.

    p_hat_t = sum phi_p (dX - b_rest h) / sum phi_p^2 h, with sums up to t.

    Returns
    -------
    ndarray, shape (N, num_steps+1)
        The estimate for each particle at each time step (0 at t=0).
    """
    _, dX, phi, b_rest = _increments(paths, pop, [param], params, means)
    estimates = np.zeros_like(paths[pop])
    estimates[:, 1:] = np.cumsum(phi[0]*(dX - b_rest*params["h"]), axis=1) / (np.cumsum(phi[0]**2, axis=1)*params["h"])
    return estimates

def learning_rate(s, lr0, lr_decay):
    """Learning rate lr0*max(s, 1)^-lr_decay at time s (a float or an array of times)."""
    return lr0*np.maximum(s, 1)**-lr_decay

def sgd_step(estimate, x, x_next, lr, pop, param, params, means):
    """One online gradient step on the log-likelihood of the increment x -> x_next.

    p <- p + lr * dL/dp, where L = (beta/2) [b (x_next - x) - b^2 h/2] is the log-likelihood of the
    step, as in :func:`likelihood_coefficients`, so dL/dp = (beta/2) phi_p(x) (x_next - x - b(x) h)
    with b evaluated at the current estimate. ``estimate``, ``x`` and ``x_next`` have the same
    shape, one entry per particle.
    """
    current = {**params, param: estimate}
    phi = basis(pop, x, current, means)[param]
    residual = x_next - x - drift(pop, x, current, means)*params["h"]
    return estimate + lr*params["beta"]/2*phi*residual

def sgd_step_shared(estimate, x, x_next, lr, N, pop, param, params, means):
    """One online gradient step of one estimate per system, with the gradient averaged over its N particles.

    ``estimate`` has shape (M,), and ``x``, ``x_next`` shape (M*N,), with system s in rows s*N to (s+1)*N - 1
    as in :func:`model.simulate`. Averaging keeps the slope of the error recursion of :func:`sgd_step` but
    divides its noise variance by N (see :func:`sgd_error_variance`).
    """
    rep = np.repeat(estimate, N)
    return estimate + (sgd_step(rep, x, x_next, lr, pop, param, params, means) - rep).reshape(-1, N).mean(axis=1)

def natural_step(estimate, info, s, x, x_next, lr, pop, names, params, means, gain=None):
    """One natural gradient step on the joint log-likelihood of the increment x -> x_next.

    p <- p + lr G^-1 dL/dp, with G the Fisher information per unit time i = (beta/2) E[phi phi^T]. The step's
    log-likelihood is c + l.p - 1/2 p.Q.p (:func:`likelihood_coefficients`), so dL/dp = l - Q p and Q is
    its observed information. The gain G is either
      - ``gain``, a fixed (k, k) matrix, e.g. the exact i from :func:`fisher_information`, or
      - if ``gain`` is None, the running average info/(1 + s + h) of the observed information, where ``info``
        should start at the identity (a prior worth one unit of time).
    ``gain = np.eye(k)`` gives plain gradient ascent, and with one name in ``names``, the step is :func:`sgd_step`.

    Preconditioning by i^-1 makes the error recursion of :func:`sgd_clt_variance` have slope I and noise
    covariance i^-1, so (p_hat_s - p)/sqrt(lr_s) -> N(0, v i^-1), with v that of :func:`sgd_clt_variance`
    for unit information (phi2 = 2/beta). With lr = 1/s (lr0 = 1, lr_decay = 1), v = 1 and
    Var(p_hat_s) = i^-1/s, the Cramer-Rao bound; the empirical gain is then close to the recursive MLE.

    Parameters
    ----------
    estimate : ndarray, shape (M, k)
        Current estimates of the parameters ``names``, one row per particle.
    info : ndarray, shape (M, k, k)
        Running observed information of each particle, updated by this step.
    s : float
        Time of x.
    x, x_next : ndarray, shape (M,)

    Returns
    -------
    estimate : ndarray, shape (M, k)
    info : ndarray, shape (M, k, k)
    """
    _, l, Q = likelihood_coefficients({pop: np.stack([x, x_next], axis=1)}, pop, names, params, means)
    info = info + Q
    G = gain if gain is not None else info/(1 + s + params["h"])
    grad = l - np.einsum("mij,mj->mi", Q, estimate)
    return estimate + lr*np.linalg.solve(G, grad[..., None])[..., 0], info

def sgd_path(paths, pop, param, params, means, initial, lr_decay, lr0=0.1):
    """Online gradient ascent on the log-likelihood, one step per time step (was OU IPS coupled online.py).

    p <- p + lr * (beta/2) * phi_p (dX - b h), i.e. lr times the gradient of the log-likelihood of
    the step (:func:`sgd_step`), with learning rate lr0*max(s, 1)^-lr_decay, where s = t*h is the
    time (not the step index), so the schedule does not depend on h.

    Parameters
    ----------
    initial : ndarray, shape (N,)
        Starting estimate for each particle.
    lr_decay : float
        Exponent of the learning rate decay, in (0, 1]. The estimate converges in mean square,
        with variance of order lr/2 for lr_decay < 1, so a small exponent converges slowly.
        See :func:`sgd_clt_variance`.
    lr0 : float, optional
        Learning rate up to time 1.

    Returns
    -------
    ndarray, shape (N, num_steps+1)
        The estimate for each particle at each time step.
    """
    X, h = paths[pop], params["h"]
    estimates = np.zeros_like(X)
    estimates[:, 0] = initial
    for t in trange(X.shape[1]-1, desc=f"Online estimate of {param}"):
        lr = learning_rate(t*h, lr0, lr_decay)
        estimates[:, t+1] = sgd_step(estimates[:, t], X[:, t], X[:, t+1], lr, pop, param, params, means)
    return estimates

def sgd_clt_variance(s, lr0, lr_decay, phi2, beta):
    """Variance of (p_hat_s - p)/sqrt(lr_s) for the online estimate of :func:`sgd_path`, at time s.

    With fixed means and the drift linear in p, the residual of a step is (p - p_hat) phi h + sigma dW,
    with sigma^2 = 2/beta, so the error e = p_hat - p of :func:`sgd_step` follows

        e <- (1 - lr (beta/2) phi^2 h) e + lr sqrt(beta/2) phi dW.

    This is a Robbins-Monro recursion whose mean-field slope and noise intensity are both
    i = (beta/2) E[phi^2], the Fisher information per unit time. Writing Var(e_s) = lr_s v_s, for large s

        v_s = 1/(2 - lr_decay/(i lr_s s)),    lr_s s = lr0 s^(1 - lr_decay).

    As s -> inf, e_s/sqrt(lr_s) -> N(0, Sigma) with
      - Sigma = 1/2 for lr_decay in (0, 1), whatever beta, lr0 and E[phi^2] (Kushner & Yin 2003, ch. 10);
      - Sigma = 1/(2 - 1/(lr0 i)) for lr_decay = 1 (cf. Sirignano & Spiliopoulos 2020), provided
        2 lr0 i > 1. Otherwise the error decays slower than sqrt(lr_s). With lr0 = 1/i, Sigma = 1 and
        Var(e_s) = 1/(i s), the Cramer-Rao bound.

    Parameters
    ----------
    s : float or ndarray
        Time(s); np.inf gives the limit Sigma.
    lr0, lr_decay : float
        As in :func:`sgd_path`.
    phi2 : float
        Stationary second moment E[phi_p(X)^2] of the basis function of the parameter.
    beta : float

    Returns
    -------
    float or ndarray
        v_s, np.inf where it is not positive.
    """
    denominator = 2 - lr_decay/(beta/2*phi2*lr0*np.maximum(s, 1)**(1 - lr_decay))
    with np.errstate(divide="ignore"):
        return np.where(denominator > 0, 1/denominator, np.inf)

def gain_clt_covariance(s, lr0, lr_decay, fisher, gain=None):
    """Covariance of (p_hat_s - p)/sqrt(lr_s) for the online estimate of :func:`natural_step` with a fixed gain G.

    The matrix version of :func:`sgd_clt_variance`. The error recursion has slope A = G^-1 i and noise
    covariance S = G^-1 i G^-1 per unit time, with i the Fisher information per unit time, so for large s the
    covariance solves the Lyapunov equation

        (A - c_s I) Sigma + Sigma (A - c_s I)^T = S,    c_s = lr_decay/(2 lr_s s) = lr_decay/(2 lr0 s^(1 - lr_decay)).

    c_s -> 0 for lr_decay < 1 and c = 1/(2 lr0) for lr_decay = 1. For G = i, Sigma = v i^-1 with v from
    :func:`sgd_clt_variance` at unit information; for G = I and lr_decay < 1, Sigma = I/2.

    Parameters
    ----------
    s : float
        Time; np.inf gives the limit.
    lr0, lr_decay : float
        As in :func:`sgd_path`.
    fisher : ndarray, shape (k, k)
        Fisher information per unit time i, e.g. :func:`fisher_information` over T, divided by T.
    gain : ndarray, shape (k, k), optional
        The gain G. If None, G = i, the limit of the empirical gain of :func:`natural_step`.

    Returns
    -------
    ndarray, shape (k, k)
        Sigma_s, all np.inf if some eigenvalue of A is at most c_s (the error then decays slower than sqrt(lr_s)).
    """
    G_inv = np.linalg.inv(fisher if gain is None else gain)
    A, S = G_inv @ fisher, G_inv @ fisher @ G_inv
    c = lr_decay/(2*lr0*np.maximum(s, 1)**(1 - lr_decay))
    if np.min(np.linalg.eigvals(A).real) <= c:
        return np.full_like(A, np.inf)
    return solve_continuous_lyapunov(A - c*np.eye(len(A)), S)

def sgd_error_variance(s, h, lr0, lr_decay, phi2, beta):
    """Variance of p_hat_s - p at time s for an online estimate started at the true value.

    Iterates the error recursion of :func:`sgd_clt_variance` with phi^2 replaced by E[phi^2],

        Var <- (1 - lr i h)^2 Var + lr^2 i h,    i = (beta/2) E[phi^2],

    from Var = 0, so unlike v_s it needs no large-s approximation. For :func:`sgd_step_shared` with
    systems of N particles, the variance is this over N.

    Parameters
    ----------
    s, h : float
        Time and time step.
    lr0, lr_decay, phi2, beta : float
        As in :func:`sgd_clt_variance`.
    """
    i = beta/2*phi2
    var = 0.0
    for k in range(int(round(s/h))):
        lr = learning_rate(k*h, lr0, lr_decay)
        var = (1 - lr*i*h)**2*var + lr**2*i*h
    return var


def likelihood_coefficients(paths, pop, names, params, means):
    """Log-likelihood of each particle's path as an exact quadratic in the parameters ``names``.

    L(p) = (beta/2) [sum b dX - 1/2 sum b^2 h] = c + l.p - 1/2 p.Q.p

    Returns
    -------
    c : ndarray, shape (N,)
    l : ndarray, shape (N, k)
    Q : ndarray, shape (N, k, k)
        k = len(names). l/Q is the MLE and Q is the (negative) Hessian.
    """
    _, dX, phi, b_rest = _increments(paths, pop, names, params, means)
    h, scale = params["h"], params["beta"]/2
    c = scale*np.sum(b_rest*dX - 0.5*b_rest**2*h, axis=1)
    l = scale*np.sum(phi*(dX - b_rest*h), axis=2).T
    Q = scale*np.einsum("inT,jnT->nij", phi, phi)*h
    return c, l, Q

def likelihood_profile(paths, pop, param, grid, params, means):
    """Likelihood of each particle at time T against values of one parameter, shape (N, len(grid))."""
    c, l, Q = likelihood_coefficients(paths, pop, [param], params, means)
    return c[:, None] + l*grid - 0.5*Q[:, 0]*grid**2

def likelihood_surface(paths, pop, names, grids, params, means):
    """Likelihood summed over particles on a grid of two parameters.

    Parameters
    ----------
    names : (str, str)
    grids : (ndarray, ndarray)
        Values of each parameter.

    Returns
    -------
    ndarray, shape (len(grids[0]), len(grids[1]))
    """
    c, l, Q = (a.sum(axis=0) for a in likelihood_coefficients(paths, pop, names, params, means))
    P = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    return c + P @ l - 0.5*np.einsum("...i,ij,...j->...", P, Q, P)

def hessian_eig(paths, pop, names, params, means):
    """Eigen-decomposition of the likelihood's curvature in the parameters ``names`` (was spectral.py).

    A zero eigenvalue means the parameters are not separately identifiable along that eigenvector.

    Returns
    -------
    e_val : ndarray, shape (k,)
        Eigenvalues in ascending order.
    e_vec : ndarray, shape (k, k)
        Eigenvectors as columns.
    """
    _, _, Q = likelihood_coefficients(paths, pop, names, params, means)
    return np.linalg.eigh(Q.sum(axis=0))


def mle_per_run(l, Q, N):
    """MLE of each run, where a run is one simulated system of N particles.

    A run's log-likelihood is the sum over its particles of c + l.p - 1/2 p.Q.p, so its MLE is
    (sum Q)^-1 sum l, using all N of its particles' paths and no data from other runs.

    Parameters
    ----------
    l, Q : ndarray
        From :func:`likelihood_coefficients`, with rows in systems of N as in :func:`model.simulate`.
    N : int

    Returns
    -------
    estimates : ndarray, shape (M, k)
        The MLE of each of the M runs.
    info : ndarray, shape (M, k, k)
        The observed information sum Q of each run.
    """
    k = l.shape[1]
    info = Q.reshape(-1, N, k, k).sum(axis=1)
    l_sum = l.reshape(-1, N, k).sum(axis=1)
    return np.linalg.solve(info, l_sum[..., None])[..., 0], info

def fisher_information(pop, names, params, means=None):
    """Expected Fisher information of the parameters ``names`` from one particle's path on [0, T].

    I_T = (beta/2) h sum_t E[phi(X_t) phi(X_t)^T], with X_t the Euler-Maruyama chain of
    :func:`model.simulate` started from N(0, ic_scale^2). The drift is affine in x and in the
    means, so X_t is Gaussian and its mean m_t and variance v_t follow exact recursions, giving
    E[phi phi^T] = phi(m_t) phi(m_t)^T + v_t s s^T with s = dphi/dx. Not valid for drifts
    that are nonlinear in x.

    Parameters
    ----------
    means : dict, optional
        Fixed means m_X, m_Y. If None, the means are E[X_t], E[Y_t] under the McKean-Vlasov
        limit law, i.e. the N -> infinity limit of the interacting particle system.

    Returns
    -------
    ndarray, shape (k, k)
        k = len(names). sqrt(N)(p_hat - p) -> N(0, I_T^-1) for the MLE from N particles.
    """
    h, sigma2 = params["h"], diffusion(params)**2
    m = {p: 0.0 for p in POPULATIONS}
    v = params["ic_scale"]**2
    info = np.zeros((len(names), len(names)))
    for _ in range(int(params["T"]/h)):
        current = means if means is not None else m
        phi_m, phi_0, phi_1 = (np.array([basis(pop, x, params, current)[p] for p in names]) for x in (m[pop], 0, 1))
        s = phi_1 - phi_0
        info += (np.outer(phi_m, phi_m) + v*np.outer(s, s))*h
        kappa = drift(pop, 0, params, current) - drift(pop, 1, params, current)
        v = (1 - kappa*h)**2*v + sigma2*h
        m = {p: m[p] + drift(p, m[p], params, current)*h for p in POPULATIONS}
    return params["beta"]/2*info
