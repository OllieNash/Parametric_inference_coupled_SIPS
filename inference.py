"""Offline MLE, online gradient estimation and the likelihood for the model in model.py.

The drift is linear in every parameter p, b = b_rest + sum_p p*phi_p, so both methods only need
phi_p(X_t) and the increments dX_t. All sums are left-point (Ito) sums.

Common arguments: ``paths`` is the output of :func:`model.simulate`, ``pop`` is "x" or "y",
and ``params``, ``means`` are as in model.py. The values of ``means`` may also be arrays that
broadcast against X_t, e.g. the empirical means from :func:`model.empirical_means`.
"""
import numpy as np
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

def sgd_path(paths, pop, param, params, means, initial, lr_decay):
    """Online gradient ascent on the log-likelihood, one step per time step (was OU IPS coupled online.py).

    p <- p + lr * sqrt(beta/2) * phi_p (dX - b h), with learning rate min(0.1, 0.1*t^-lr_decay).

    Parameters
    ----------
    initial : ndarray, shape (N,)
        Starting estimate for each particle.
    lr_decay : float
        Exponent of the learning rate decay.

    Returns
    -------
    ndarray, shape (N, num_steps+1)
        The estimate for each particle at each time step.
    """
    X, h = paths[pop], params["h"]
    estimates = np.zeros_like(X)
    estimates[:, 0] = initial
    for t in trange(X.shape[1]-1, desc=f"Online estimate of {param}"):
        lr = 0.1*max(t, 1)**-lr_decay
        current = {**params, param: estimates[:, t]}
        phi = basis(pop, X[:, t], current, means)[param]
        residual = X[:, t+1] - X[:, t] - drift(pop, X[:, t], current, means)*h
        estimates[:, t+1] = estimates[:, t] + lr*np.sqrt(params["beta"]/2)*phi*residual
    return estimates


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
