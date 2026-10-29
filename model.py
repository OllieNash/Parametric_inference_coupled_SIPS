"""Coupled two-population McKean-Vlasov O-U system: parameters, drift, simulation and presets.

For population X::

    dX = [theta_1*(mu_x - X) + (alpha_1/2)*(m_X - X) + (gamma/2)*(m_Y - X)] dt + sqrt(2/beta) dW

and the same for Y with theta_2, alpha_2, mu_y. m_X, m_Y are the population means. Every other
model in this repo is a special case of this one (see PRESETS).

Throughout, ``pop`` is a population name ("x" or "y"), ``params`` a dict like PARAMS, and
``means`` a dict {"x": m_X, "y": m_Y}.
"""
import numpy as np
from tqdm import trange

#Parameters & inputs
PARAMS = dict(
    N = 1, #Number of particles
    T = 300, #Time
    h = 0.01, #time step
    theta_1 = 1, #Mean reversion parameter for X
    theta_2 = 1.5, #Mean reversion parameter for Y
    alpha_1 = 1.1, #interaction strength for X
    alpha_2 = 1, #interaction strength for Y
    gamma = 1, #cross-interaction strength
    mu_x = 0, #true mean for x
    mu_y = 0, #true mean for y
    beta = 5, #diffusion coefficient
    ic_scale = 0.45, #std of the initial distribution
)

#(confining parameter, interaction parameter, true mean, other population) for each population
POPULATIONS = {
    "x": ("theta_1", "alpha_1", "mu_x", "y"),
    "y": ("theta_2", "alpha_2", "mu_y", "x"),
}

#Special cases: parameter overrides, the (population, parameter) pairs to estimate, and
#how to get the means m_X, m_Y ("ips" simulates the full particle system, otherwise they are 0)
PRESETS = {
    #Plain O-U, no interaction: estimate theta (was O-U ND.py)
    "ou": dict(params=dict(N=500, T=100, alpha_1=0, gamma=0, beta=1, ic_scale=0.2),
               estimate=[("x", "theta_1")]),
    #Single M-V O-U population: estimate the interaction (was O-U IPS 1D vec.py).
    #alpha_1 = 2*alpha there, since the drift here uses alpha_1/2
    "mv_ou": dict(params=dict(N=100, T=100, alpha_1=2, gamma=0, beta=10, ic_scale=0.5),
                  estimate=[("x", "alpha_1")]),
    #Coupled populations: cross-interaction from X, confining from Y (was O-U IPS coupled 1D .py)
    "coupled": dict(params=dict(), estimate=[("x", "gamma"), ("y", "theta_2")],
                    ips=dict(N=500, T=100)),
}


def basis(pop, x, params, means):
    """Derivative of the drift of ``pop`` with respect to each of its parameters.

    The drift is linear in every parameter, so drift = sum(params[p] * phi_p).

    Parameters
    ----------
    pop : {"x", "y"}
    x : ndarray
        Particle positions, any shape.
    params, means : dict

    Returns
    -------
    dict
        {parameter name: phi_p(x)} for the confining, interaction and cross-interaction
        parameters of ``pop``, each the same shape as ``x``.
    """
    theta, alpha, mu, other = POPULATIONS[pop]
    return {
        theta: params[mu] - x,
        alpha: (means[pop] - x)/2,
        "gamma": (means[other] - x)/2,
    }

def drift(pop, x, params, means):
    """Drift of population ``pop`` at positions ``x`` (same shape as ``x``)."""
    return sum(params[p]*phi for p, phi in basis(pop, x, params, means).items())

def diffusion(params):
    """Diffusion coefficient sqrt(2/beta)."""
    return np.sqrt(2/params["beta"])


def simulate(params, means=None):
    """Simulate X and Y with Euler-Maruyama.

    Parameters
    ----------
    params : dict
        Uses N, T, h and ic_scale as well as the drift parameters.
    means : dict, optional
        Fixed means m_X, m_Y for the interactions. If None, each step uses the empirical
        means of the particles instead, i.e. the full interacting particle system.

    Returns
    -------
    dict
        {"x": X, "y": Y}, each of shape (N, num_steps+1).
    """
    N, h = params["N"], params["h"]
    num_steps = int(params["T"]/h)
    paths = {}
    for pop in POPULATIONS:
        paths[pop] = np.zeros((N, num_steps+1))
        paths[pop][:, 0] = np.random.normal(0, params["ic_scale"], size=N)

    for t in trange(num_steps, desc="Simulating"):
        m = means if means is not None else {pop: np.mean(paths[pop][:, t]) for pop in POPULATIONS}
        for pop, X in paths.items():
            dW = np.random.normal(loc=0, scale=np.sqrt(h), size=N)
            X[:, t+1] = X[:, t] + drift(pop, X[:, t], params, m)*h + diffusion(params)*dW
    return paths

def ips_means(params, N, T):
    """Means m_X, m_Y at time T from the full IPS with N particles (was IPS_generation.py).

    Returns
    -------
    dict
        {"x": m_X, "y": m_Y}, to pass as ``means`` to the other functions.
    """
    paths = simulate({**params, "N": N, "T": T})
    return {pop: np.mean(X[:, -1]) for pop, X in paths.items()}

def setup(preset, **overrides):
    """Build everything a driver needs for one of PRESETS.

    Parameters
    ----------
    preset : str
        A key of PRESETS.
    **overrides
        Values replacing entries of PARAMS, e.g. ``T=50``.

    Returns
    -------
    params : dict
    targets : list of (population, parameter name)
        The parameters to estimate.
    means : dict
        From :func:`ips_means` if the preset asks for it, otherwise all 0.
    """
    p = PRESETS[preset]
    params = {**PARAMS, **p["params"], **overrides}
    means = ips_means(params, **p["ips"]) if "ips" in p else {pop: 0 for pop in POPULATIONS}
    return params, p["estimate"], means
