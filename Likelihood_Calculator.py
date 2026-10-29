import numpy as np
from tqdm import tqdm

def likelihood_x_over_gamma(test_gamma, X_slice, X_delta, theta_1, alpha_1, mu_x, beta, h):
    tg = np.asarray(test_gamma)[None, None, :]  # (1,1,G)
    return -np.sqrt(beta/2) * (
        np.sum(
            (theta_1*(mu_x - X_slice[:, 1:, None]) + (alpha_1/2)*(0 - X_slice[:, 1:, None]) + (tg/2)*(0 - X_slice[:, 1:, None]))
            * X_delta[:, :, None], axis=1
        )
        + 0.5 * np.sum(
            (theta_1*(mu_x - X_slice[:, 1:, None]) + (alpha_1/2)*(0 - X_slice[:, 1:, None]) + (tg/2)*(0 - X_slice[:, 1:, None]))**2,
            axis=1
        ) * h
    ) * np.sqrt(beta/2)  # -> shape: (N, len(test_gamma))

def likelihood_y_over_theta(test_theta, Y_slice, Y_delta, alpha_2, mu_y, gamma, beta, h):
    tt = np.asarray(test_theta)[None, None, :]   # (1,1,Tθ)
    return -np.sqrt(beta/2) * (
        np.sum(
            (tt*(mu_y - Y_slice[:, 1:, None]) + (alpha_2/2)*(0 - Y_slice[:, 1:, None]) + (gamma/2)*(0 - Y_slice[:, 1:, None]))
            * Y_delta[:, :, None], axis=1
        )
        + 0.5 * np.sum(
            (tt*(mu_y - Y_slice[:, 1:, None]) + (alpha_2/2)*(0 - Y_slice[:, 1:, None]) + (gamma/2)*(0 - Y_slice[:, 1:, None]))**2,
            axis=1
        ) * h
    ) * np.sqrt(beta/2)  # -> shape: (N, len(test_theta))

def y_likelihood_surface(test_theta, test_gamma, Y_slice, Y_delta, alpha_2, mu_y, beta, h):
    print("hi")
    Y_diff = 0 - Y_slice[:, 1:]     # shape (N, T-1)
    A = mu_y - Y_slice[:, 1:]       # shape (N, T-1)
    sqrtb = np.sqrt(beta / 2)

    Tθ = len(test_theta)
    G  = len(test_gamma)

    L = np.zeros((Tθ, G))  # final output (Tθ, G)

    for i, theta in enumerate(tqdm(test_theta, desc="Calculating likelihood surface")):
        d_theta = theta * A + (alpha_2/2) * Y_diff    # (N, T-1)

        for j, gamma in enumerate(test_gamma):
            d = d_theta + (gamma/2) * Y_diff          # (N, T-1)

            term = np.sum(d * Y_delta, axis=1) + 0.5 * np.sum(d**2, axis=1) * h
            L[i, j] = -sqrtb * term.sum() * sqrtb

    return L
