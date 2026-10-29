"""Offline MLE, online gradient estimation and the likelihood for the model in model.py.

The drift is linear in every parameter p, b = b_rest + sum_p p*phi_p, so both methods only need
phi_p(X_t) and the increments dX_t. All sums are left-point (Ito) sums.

Common arguments: ``paths`` is the output of :func:`model.simulate`, ``pop`` is "x" or "y",
and ``params``, ``means`` are as in model.py.
"""
import numpy as np
from tqdm import trange

from model import basis, drift


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
