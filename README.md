# Inference for coupled McKean–Vlasov O-U systems

This is an implementation of inferring the drift parameters of a coupled two-population McKean–Vlasov Ornstein–Uhlenbeck system by maximum likelihood (offline, closed form) and by online stochastic gradient ascent.

For population X:

    dX = [θ₁(μ_x − X) + (α₁/2)(m_X − X) + (γ/2)(m_Y − X)] dt + √(2/β) dW

Y follows the same equation with θ₂, α₂, μ_y. Here m_X and m_Y are the population means, and γ is the cross-interaction.

The drift is linear in every parameter, so both methods need only ∂b/∂p and the increments dX:

- MLE: p̂ = Σ φ_p (dX − b₋ₚ h) / Σ φ_p² h
- Gradient step: p ← p + lr·√(β/2)·φ_p (dX − b h)
- Log-likelihood: exactly quadratic in the parameters.

## Files

| File | Contents |
|---|---|
| `model.py` | Parameters, drift, Euler–Maruyama simulation, IPS means, and the presets below |
| `inference.py` | Running MLE, online gradient estimate, likelihood profile/surface, Hessian eigenvectors |
| `plotter.py` | Plots |
| `run_mle.py` | Offline MLE driver |
| `run_sgd.py` | Online gradient driver |
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

When all the means are 0, the (θ₂, γ) likelihood for Y is a ridge: only θ₂ + γ/2 is identified, and one Hessian eigenvalue is 0. Setting a non-zero `mu_x` or `mu_y` separates them.
