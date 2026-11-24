# Inference for coupled McKean–Vlasov O-U systems

This is an implementation of inferring the drift parameters of a coupled two-population McKean–Vlasov Ornstein–Uhlenbeck system by maximum likelihood (offline, closed form) and by online stochastic gradient ascent.

For population X:

    dX = [θ₁(c_x − X) + (α₁/2)(m_X − X) + (γ/2)(m_Y − X)] dt + √(2/β) dW

Y follows the same equation with θ₂, α₂, c_y. Here m_X and m_Y are the population means, and γ is the cross-interaction.

c_x and c_y are the centres of the confining potentials θ(x − c)²/2: the fixed points each population is pulled towards. They are not in general the true means. The cross-interaction pulls the two means towards each other, and the stationary means equal the centres only when γ = 0 or c_x = c_y. The `ou` and `mv_ou` presets are in that special case.

The drift is linear in every parameter, so both methods need only ∂b/∂p and the increments dX:

- MLE: p̂ = Σ φ_p (dX − b₋ₚ h) / Σ φ_p² h
- Gradient step: p ← p + lr·(β/2)·φ_p (dX − b h), i.e. lr times the gradient of the step's log-likelihood, with lr = lr₀·max(t, 1)^−d for t the time; d ∈ (0, 1]; the error is of order √lr, so a small d converges slowly
- Step with a matrix gain, for one parameter or several jointly: p ← p + lr·G⁻¹∇L (`inference.natural_step`). For the natural gradient, G is either the running empirical Fisher information or the exact one, i = (β/2)E[φφᵀ]. G = I is plain gradient ascent
- Log-likelihood: exactly quadratic in the parameters.

## Files

| File | Contents |
|---|---|
| `model.py` | Parameters, drift, Euler–Maruyama simulation (one or many independent systems), IPS means, and the presets below |
| `inference.py` | Running MLE, online gradient estimate, likelihood profile/surface, Hessian eigenvectors, MLE per run, Fisher information |
| `plotter.py` | Plots |
| `run_mle.py` | Offline MLE driver |
| `run_sgd.py` | Online gradient driver: one long path, plain or natural gradient, joint or individual |
| `validate_mle.py` | Monte Carlo validation of the MLE as N grows: consistency, RMSE against Cramér–Rao, Wald coverage, asymptotic normality |
| `validate_sgd.py` | Monte Carlo validation of the online estimate as t grows (consistency and the CLT for (θ̂ₜ − θ₀)/√lrₜ), with `N`, its error against N at fixed T, or with `natural` or `sgd`, the natural or plain gradient estimate of (θ₂, γ), joint or individual |
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
    python validate_mle.py               # coupled
    python validate_mle.py mv_ou         # or ou

If `python` is not on your PATH, activate the virtual environment first (`source .venv/bin/activate`) or call `.venv/bin/python`. Figures open one at a time: close each to get the next.

### Online gradient estimation: the four cases

Both online scripts take the same two arguments:

    python run_sgd.py      {sgd | natural} [joint | individual]
    python validate_sgd.py {sgd | natural} [joint | individual]

- The first argument is the gradient update:
  - `sgd` is the plain gradient, p ← p + lr·∇L.
  - `natural` is the natural gradient, p ← p + lr·G⁻¹∇L. It always runs two gains side by side: the empirical one (running observed Fisher information) and the exact one (G = i).
- The second argument is optional and defaults to `joint`:
  - `joint` estimates (θ₂, γ) together from Y.
  - `individual` estimates θ₂ and γ each alone from Y, with the other at its true value.

| Case | `run_sgd.py` (one long path, T = 50000) | `validate_sgd.py` (M = 2000 runs, T = `params["T"]`) |
|---|---|---|
| Plain gradient, joint | `python run_sgd.py sgd` | `python validate_sgd.py sgd` |
| Plain gradient, individual | `python run_sgd.py sgd individual` | `python validate_sgd.py sgd individual` |
| Natural gradient, joint | `python run_sgd.py natural` | `python validate_sgd.py natural` |
| Natural gradient, individual | `python run_sgd.py natural individual` | `python validate_sgd.py natural individual` |

**What the scripts produce**

- `run_sgd.py` simulates one path with the means fixed. For each estimate (each gain, and each parameter or the joint pair), it prints the final estimate and plots the estimate and its L2 error against time.
- `validate_sgd.py` prints, at the CLT times, the bias, sd, RMSE, CLT sd, Cramér–Rao sd and 95% coverage. It plots the consistency, error and CLT-variance summary for each parameter. For `joint`, it also plots the 95% ellipse and a χ² QQ plot at t = T. The theory is in "Natural and plain gradient, joint or individual" below.

**Settings** (initial learning rate lr₀ and decay d) are in `GAIN_STUDY` at the top of each script:

| Update | `run_sgd.py` | `validate_sgd.py` |
|---|---|---|
| `natural` | lr₀ = 1, d = 1, i.e. lr = 1/t, the efficient (Cramér–Rao) rate | lr₀ = 1, d = 1 |
| `sgd` | lr₀ = 2.5, d = 0.9 | lr₀ = 2.5, d = 0.75 |

**Run time.** A natural-gradient step takes about 90 µs, and a plain one about 20 µs. In `run_sgd.py`, at T = 50000 (5 million steps), each estimate takes roughly 8 minutes, and there are:

| Case | Estimates | Time |
|---|---|---|
| `natural` | 2 (one per gain) | about 15 minutes |
| `natural individual` | 4 | about 30 minutes |
| `sgd` | 1 | about 8 minutes |
| `sgd individual` | 2 | about 15 minutes |

For a quick look, lower `"T"` in `params`. `validate_sgd.py` updates all M runs at once, so its time grows with `params["T"]` rather than with M.

**The other modes**

- `python run_sgd.py` with no arguments is the original scalar study: θ₂ from Y and γ from X, each alone by plain gradient, with likelihood profile plots.
- `validate_sgd.py` also keeps:
  - `python validate_sgd.py` (or `time`): the same scalar study, as t grows.
  - `python validate_sgd.py N`: its error against N at fixed T.

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

## Validation of the online estimate (`validate_sgd.py`)

M independent runs each simulate one particle with the means fixed, as in `run_sgd.py`. Each run updates its own online estimate at every step with learning rate lrₜ = lr₀·max(t, 1)^−d. The script keeps only the current positions and estimates, so memory does not grow with T, and records the estimates on a grid of times.

With fixed means and a drift linear in the parameter, the error e = θ̂ − θ₀ follows e ← (1 − lr·(β/2)·φ²h)e + lr·√(β/2)·φ·ΔW. This is a Robbins–Monro recursion whose slope and noise intensity are both i = (β/2)E[φ²], the Fisher information per unit time. So

- for d ∈ (0, 1), (θ̂ₜ − θ₀)/√lrₜ → N(0, Σ) with Σ = ½. Σ is the same for every parameter, whatever β, lr₀ and E[φ²] (Kushner & Yin 2003, ch. 10);
- for d = 1, Σ = 1/(2 − 1/(lr₀·i)), provided 2·lr₀·i > 1 (cf. Sirignano & Spiliopoulos 2020). With lr₀ = 1/i, Σ = 1 and Var(θ̂ₜ) = 1/(i·t), the Cramér–Rao bound.

`inference.sgd_clt_variance` also gives the finite-time variance vₜ = 1/(2 − d/(i·lr₀t^(1−d))). It approaches Σ only like t^−(1−d), so the script uses d = 0.75 rather than the 0.9 of `run_sgd.py`. At finite t there is also a small O(lrₜ) bias, from the correlation of φ(Xₜ) with past noise. Both effects vanish relative to √lrₜ.

The plots show:

- **Consistency:** the runs' θ̂ₜ against t, with θ₀ ± 1.96√(Σ·lrₜ).
- **Error:** RMSE, sd and |bias| against the CLT standard deviation √(Σ·lrₜ) ∝ t^−d/2.
- **CLT variance:** the variance over the runs of (θ̂ₜ − θ₀)/√lrₜ against t, with Σ and vₜ.
- **CLT:** at t = 10, 100, 1000 and 5000, a histogram of (θ̂ₜ − θ₀)/√lrₜ against N(0, Σ), with θ̂ₜ itself on the top axis, plus normal QQ plots.

### Error against N (`python validate_sgd.py N`)

This is the online counterpart of `validate_mle.py`. For each N in a grid, M independent runs each simulate their own IPS of N particles on [0, T] (T = 50), with the run's empirical means in the interactions. Each run keeps one online estimate, updated at every step with the gradient averaged over its N particles (`inference.sgd_step_shared`). Averaging keeps the slope of the error recursion but divides its noise variance by N, so Var(θ̂_T − θ₀) ≈ V_T/N. Here V_T is the variance of the single-particle recursion started at θ₀ (`inference.sgd_error_variance`).

The estimates start at θ₀. The transient from a random start decays like exp(−i∫lr) whatever N, and would put a floor under the error at large N. The figure shows RMSE, sd and |bias| against N on log-log axes, with the theory line √(V_T/N) and the fitted slope of the RMSE, which should be close to −½.

### Natural and plain gradient, joint or individual (`python validate_sgd.py {natural|sgd} [joint|individual]`)

This uses the same setup as the time study, but each run estimates (θ₂, γ) from Y with p ← p + lr·G⁻¹∇L (`inference.natural_step`). The gradient and the step's observed information come from `inference.likelihood_coefficients` on the one-step path. With `joint` (the default) the two parameters are estimated together. With `individual`, each is estimated alone with the other known. The settings of each update are in `GAIN_STUDY`.

`natural` runs two gains side by side from the same starting estimates:

- **empirical**: G is the running average of the observed information, started at the identity (a prior worth one unit of time). It needs no knowledge of θ₀.
- **exact**: G = i = (β/2)E[φφᵀ] from `inference.fisher_information`, an oracle.

`sgd` uses G = I, the plain gradient. Run individually, this is the scalar update of the time study.

With a fixed gain G, the error recursion has slope A = G⁻¹i and noise covariance S = G⁻¹iG⁻¹. So (θ̂ₜ − θ₀)/√lrₜ → N(0, Σ), where Σ solves the Lyapunov equation (A − cI)Σ + Σ(A − cI)ᵀ = S, with c = 0 for d < 1 and c = 1/(2lr₀) for d = 1 (`inference.gain_clt_covariance`, the matrix version of `sgd_clt_variance`). The cases are:

- **Plain gradient:** Σ = I/2 for d < 1, as in the scalar case. The approach to it is slow along the eigenvector of i with the small eigenvalue (≈ 0.28 here), which is the direction the ridge of the likelihood surface lies along.
- **Natural gradient:** Σ = v·i⁻¹, with v = 1/2 for d < 1 and v = 1/(2 − 1/lr₀) for d = 1. The script uses lr = 1/t (lr₀ = 1, d = 1), so Σ = i⁻¹ and Var(θ̂ₜ) = i⁻¹/t, the Cramér–Rao bound. With this rate the empirical gain is essentially the recursive MLE.

The Cramér–Rao bound depends on what is known. Estimated jointly, the bound is (i⁻¹)ⱼⱼ/t. Estimated alone, it is 1/(iⱼⱼt), which is smaller because θ₂ and γ are correlated.

The two gains have the same limit, but their finite-t biases differ. Both come from the correlation of φ(Xᵤ) with past noise, so they appear even when the estimates start at θ₀:

- With the exact gain, the bias accumulates to about 3·log t/t (≈ 0.02 at t = 1000, about half the Cramér–Rao sd), which lowers the coverage.
- The empirical gain normalises by the information actually observed, which cancels most of this. Its bias is O(1/t), like the MLE's.

The script prints, for each gain and parameter, the bias, sd and RMSE against the CLT and Cramér–Rao sd, and the coverage. It plots the consistency and CLT summary for each. For joint estimation it also plots the 95% ellipse and χ² QQ plot at t = T.
