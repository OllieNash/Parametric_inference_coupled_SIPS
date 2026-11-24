"""Coupled two-population McKean-Vlasov O-U system: parameters, drift, simulation and presets.

For population X::

    dX = [theta_1*(c_x - X) + (alpha_1/2)*(m_X - X) + (gamma/2)*(m_Y - X)] dt + sqrt(2/beta) dW

and the same for Y with theta_2, alpha_2, c_y. c_x, c_y are the centres of the confining
potentials theta*(x - c)^2/2, the points each population is pulled towards, and m_X, m_Y are the
population means. The centres are the true means only when gamma = 0 or c_x = c_y. Every other
model in this repo is a special case of this one (see PRESETS).

Throughout, ``pop`` is a population name ("x" or "y"), ``params`` a dict like PARAMS, and
``means`` a dict {"x": m_X, "y": m_Y}.
"""
import numpy as np
from tqdm import trange

#Parameters & inputs
PARAMS = dict(
    N = 1, #Number of particles
    T = 500, #Time
    h = 0.01, #time step
    theta_1 = 1, #Mean reversion parameter for X
    theta_2 = 1, #Mean reversion parameter for Y
    alpha_1 = 1, #interaction strength for X
    alpha_2 = 1, #interaction strength for Y
    gamma = 1, #cross-interaction strength
    c_x = 2, #centre of the confining potential for X
    c_y = 0, #centre of the confining potential for Y
    beta = 8, #diffusion coefficient
    ic_scale = 0.001, #std of the initial distribution
)

#(confining parameter, interaction parameter, confining centre, other population) for each population
POPULATIONS = {
    "x": ("theta_1", "alpha_1", "c_x", "y"),
    "y": ("theta_2", "alpha_2", "c_y", "x"),
}

#Special cases: parameter overrides, the (population, parameter) pairs to estimate, and
#how to get the means m_X, m_Y ("ips" simulates the full particle system, otherwise they are 0)
PRESETS = {
    #Plain O-U, no interaction: estimate theta (was O-U ND.py)
    "ou": dict(params=dict(N=500, T=100, c_x=0, alpha_1=0, gamma=0, beta=1, ic_scale=0.2),
               estimate=[("x", "theta_1")]),
    #Single M-V O-U population: estimate the interaction (was O-U IPS 1D vec.py).
    #alpha_1 = 2*alpha there, since the drift here uses alpha_1/2
    "mv_ou": dict(params=dict(N=100, T=100, c_x=0, alpha_1=2, gamma=0, beta=10, ic_scale=0.5),
                  estimate=[("x", "alpha_1")]),
    #Coupled populations: cross-interaction from X, confining from Y (was O-U IPS coupled 1D .py).
    "coupled": dict(params=dict(c_x=2), estimate=[("x", "gamma"), ("y", "theta_2")],
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
    theta, alpha, centre, other = POPULATIONS[pop]
    return {
        theta: params[centre] - x,
        alpha: (means[pop] - x)/2,
        "gamma": (means[other] - x)/2,
    }

def drift(pop, x, params, means):
    """Drift of population ``pop`` at positions ``x`` (same shape as ``x``)."""
    return sum(params[p]*phi for p, phi in basis(pop, x, params, means).items())

def diffusion(params):
    """Diffusion coefficient sqrt(2/beta)."""
    return np.sqrt(2/params["beta"])

def em_step(pop, x, params, means):
    """One Euler-Maruyama step of population ``pop`` from positions ``x``, with ``means`` held over the step."""
    h = params["h"]
    dW = np.random.normal(loc=0, scale=np.sqrt(h), size=np.shape(x))
    return x + drift(pop, x, params, means)*h + diffusion(params)*dW


def simulate(params, means=None, systems=1, final_only=False):
    """Simulate X and Y with Euler-Maruyama.

    Parameters
    ----------
    params : dict
        Uses N, T, h and ic_scale as well as the drift parameters.
    means : dict, optional
        Fixed means m_X, m_Y for the interactions. If None, each step uses the empirical
        means of the particles instead, i.e. the full interacting particle system.
    systems : int, optional
        Number of independent copies of the N-particle system. Rows s*N to (s+1)*N - 1 are
        system s, and with ``means`` None each system uses its own empirical means.
    final_only : bool, optional
        If True, keep only the positions at time T instead of the whole paths, so memory
        does not grow with the number of steps.

    Returns
    -------
    dict
        {"x": X, "y": Y}, each of shape (systems*N, num_steps+1), or (systems*N,) at time T
        if ``final_only``.
    """
    N, h = params["N"], params["h"]
    rows = systems*N
    num_steps = int(params["T"]/h)
    state = {pop: np.random.normal(0, params["ic_scale"], size=rows) for pop in POPULATIONS}
    if not final_only:
        paths = {pop: np.zeros((rows, num_steps+1)) for pop in POPULATIONS}
        for pop, X in state.items():
            paths[pop][:, 0] = X

    for t in trange(num_steps, desc="Simulating"):
        m = means if means is not None else {pop: _system_means(X, N) for pop, X in state.items()}
        for pop, X in state.items():
            state[pop] = em_step(pop, X, params, m)
            if not final_only:
                paths[pop][:, t+1] = state[pop]
    return state if final_only else paths

def _system_means(X, N):
    """Mean over each system of N consecutive rows of ``X``, repeated back to the shape of ``X``."""
    return np.repeat(X.reshape(-1, N, *X.shape[1:]).mean(axis=1), N, axis=0)

def empirical_means(paths, N):
    """Empirical means of each system at the left-point times, to pass as ``means`` to inference.py.

    Parameters
    ----------
    paths : dict
        From :func:`simulate`, with systems of N particles.

    Returns
    -------
    dict
        {"x": m_X, "y": m_Y}, each of shape (systems*N, num_steps): row i holds the means of the
        system particle i belongs to, at every time step but the last.
    """
    return {pop: _system_means(X[:, :-1], N) for pop, X in paths.items()}

def ips_means(params, N, T):
    """Means m_X, m_Y at time T from the full IPS with N particles (was IPS_generation.py).

    Returns
    -------
    dict
        {"x": m_X, "y": m_Y}, to pass as ``means`` to the other functions.
    """
    final = simulate({**params, "N": N, "T": T}, final_only=True)
    return {pop: np.mean(X) for pop, X in final.items()}

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
