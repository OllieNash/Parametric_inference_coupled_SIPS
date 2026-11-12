# Inference for coupled McKean–Vlasov O-U systems

This is an implementation of inferring the drift parameters of a coupled two-population McKean–Vlasov Ornstein–Uhlenbeck system by maximum likelihood (offline, closed form) and by online stochastic gradient ascent.

For population X:

    dX = [θ₁(c_x − X) + (α₁/2)(m_X − X) + (γ/2)(m_Y − X)] dt + √(2/β) dW

Y follows the same equation with θ₂, α₂, c_y. Here m_X and m_Y are the population means, and γ is the cross-interaction.

c_x and c_y are the centres of the confining potentials θ(x − c)²/2: the fixed points each population is pulled towards. They are not in general the true means. The cross-interaction pulls the two means towards each other, and the stationary means equal the centres only when γ = 0 or c_x = c_y. The `ou` and `mv_ou` presets are in that special case.

The drift is linear in every parameter, so both methods need only ∂b/∂p and the increments dX:

- MLE: p̂ = Σ φ_p (dX − b₋ₚ h) / Σ φ_p² h
- Gradient step: p ← p + lr·√(β/2)·φ_p (dX − b h)
- Log-likelihood: exactly quadratic in the parameters.

## Files

| File | Contents |
|---|---|
| `model.py` | Parameters, drift, Euler–Maruyama simulation (one or many independent systems), IPS means, and the presets below |
| `inference.py` | Running MLE, online gradient estimate, likelihood profile/surface, Hessian eigenvectors, MLE per run, Fisher information |
| `plotter.py` | Plots |
| `run_mle.py` | Offline MLE driver |
| `run_sgd.py` | Online gradient driver |
| `validate_mle.py` | Monte Carlo validation of the MLE as N grows: consistency, RMSE against Cramér–Rao, Wald coverage, asymptotic normality |
| `non_markovian.py` | Separate experiment: S-L/double-well drift with averaged noise |

## Presets (`model.PRESETS`)

| Preset | Model | Estimates |
|---|---|---|
| `ou` | Plain O-U (α₁ = γ = 0) | θ₁ |
| `mv_ou` | One M-V population (γ = 0) | α₁ (= 2α in the single-population drift) |
| `coupled` | Both populations, means from the full IPS | γ from X, θ₂ from Y, plus the (θ₂, γ) likelihood surface |

## Running

    pip install -r requirements.txt
    python run_mle.py            # coupled
    python run_mle.py ou         # or mv_ou
    python run_sgd.py
    python validate_mle.py               # coupled
    python validate_mle.py mv_ou         # or ou

## Validation (`validate_mle.py`)

For each N in a grid, the script does M independent runs. Each run simulates its own interacting particle system (IPS) of N particles on a fixed [0, T] and gives one MLE, θ̂ = (Σᵢ Qᵢ)⁻¹ Σᵢ lᵢ. This maximises that run's log-likelihood, which is a sum over its N particles (`inference.mle_per_run`). Every plot summarises the M per-run MLEs at each N.

- The confining terms use the centres c_x and c_y from the parameters.
- The interaction terms use the run's empirical means m_x and m_y at each time step (`model.empirical_means`).

Theory says √N(θ̂ − θ₀) → N(0, I_T⁻¹), where I_T is the Fisher information of one particle's path on [0, T] in the N → ∞ (McKean–Vlasov) limit (Kasonga 1990). `inference.fisher_information` computes I_T exactly for the Euler–Maruyama chain. X_t is Gaussian here, so its mean and variance follow simple recursions. This holds for any drift affine in x and the means, which covers every preset.

The plots show:

- **Consistency:** the runs' θ̂ against N.
- **RMSE:** RMSE, sd and |bias| over the runs, against the Cramér–Rao line √(I_T⁻¹/N).
- **Wald coverage:** the fraction of runs whose interval θ̂ ± 1.96√((ΣQ)⁻¹) contains θ₀.
- **Normality:** for each N, a histogram of the runs' √N(θ̂ − θ₀) against N(0, I_T⁻¹), with θ̂ itself on the top axis, plus QQ plots.
- **Joint (θ₂, γ) for `coupled`:** one point per run with the 95% ellipse, and a χ² QQ plot of the Mahalanobis distances.

θ₂ and γ are separately identifiable from Y only if c_y ≠ m_X. At stationarity this holds exactly when c_x ≠ c_y. If c_x = c_y, whether 0 or not, the (θ₂, γ) likelihood for Y is a ridge: only θ₂ + γ/2 is identified, and one Hessian eigenvalue is 0. For this reason the `coupled` preset uses c_x = 2 and c_y = 0.
