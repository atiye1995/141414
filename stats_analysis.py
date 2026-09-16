"""
STATISTICAL ANALYSIS - generic, works for any system given (q_true_table,
coeffs, E_fci). Implements: linear-only, bootstrap-exponential, BMA-Laplace
(flat prior), and the spike-and-slab remedy.
"""
import numpy as np
from scipy.stats import gamma as gamma_dist, norm

lambdas = np.array([0.5, 1.0, 1.5])


def E_of_q_group(q_group, coeffs):
    return np.tensordot(2 * q_group - 1, coeffs, axes=([-1], [0]))


# ---------------- linear-only ----------------
def fit_linear_wls(E_obs, sigma, lam):
    X = np.stack([np.ones_like(lam), lam], axis=1)
    W = np.diag(1 / sigma ** 2)
    XtWX = X.T @ W @ X
    theta = np.linalg.solve(XtWX, X.T @ W @ E_obs)
    resid = E_obs - X @ theta
    logL = -0.5 * np.sum((resid / sigma) ** 2) - np.sum(np.log(sigma)) - 1.5 * np.log(2 * np.pi)
    Sigma_post = np.linalg.inv(XtWX)
    return theta, Sigma_post, logL


def linear_only_interval(counts, n_shots, coeffs):
    q_mle = counts / n_shots
    E_obs = np.array([E_of_q_group(q_mle[i], coeffs) for i in range(3)])
    var_q = q_mle * (1 - q_mle) / n_shots
    var_E = np.array([np.sum(coeffs ** 2 * 4 * var_q[i]) for i in range(3)])
    sigma_obs = np.sqrt(np.maximum(var_E, 1e-12))
    theta, Sigma_post, logL = fit_linear_wls(E_obs, sigma_obs, lambdas)
    A_hat, se = theta[0], np.sqrt(Sigma_post[0, 0])
    return A_hat - 1.96 * se, A_hat + 1.96 * se


# ---------------- bootstrap-exponential ----------------
def exp_extrapolate(E1, E2, E3):
    denom1 = E2 - E1
    valid = np.abs(denom1) > 1e-12
    r = np.full_like(E1, np.nan)
    r[valid] = (E3[valid] - E2[valid]) / denom1[valid]
    valid = valid & (r > 0) & np.isfinite(r)
    F = np.full_like(E1, np.nan)
    F[valid] = E1[valid] - denom1[valid] / r[valid]
    return F, valid


def bootstrap_interval(counts, n_shots, coeffs, rng, n_boot=8000):
    q_mle = counts / n_shots
    boot_counts = rng.binomial(n_shots, q_mle[None, :, :], size=(n_boot, 3, len(coeffs)))
    boot_q = boot_counts / n_shots
    E1b = E_of_q_group(boot_q[:, 0, :], coeffs)
    E2b = E_of_q_group(boot_q[:, 1, :], coeffs)
    E3b = E_of_q_group(boot_q[:, 2, :], coeffs)
    F_all, valid = exp_extrapolate(E1b, E2b, E3b)
    F_valid = F_all[valid]
    if len(F_valid) < 50:
        return None
    lo, hi = np.percentile(F_valid, [2.5, 97.5])
    return lo, hi


# ---------------- BMA-Laplace (flat prior, continuous) ----------------
def fit_quadratic_exact(E_obs, sigma, lam):
    X = np.stack([np.ones_like(lam), lam, lam ** 2], axis=1)
    theta = np.linalg.solve(X, E_obs)
    logL = -np.sum(np.log(sigma)) - 1.5 * np.log(2 * np.pi)
    Xinv = np.linalg.inv(X)
    Sigma_post = Xinv @ np.diag(sigma ** 2) @ Xinv.T
    return theta, Sigma_post, logL


def fit_exponential_exact(E_obs, sigma, lam):
    E1, E2, E3 = E_obs
    denom1 = E2 - E1
    if abs(denom1) < 1e-14:
        return None
    r = (E3 - E2) / denom1
    if r <= 0 or not np.isfinite(r):
        return None
    d = lam[1] - lam[0]
    c_hat = -np.log(r) / d
    x1 = np.exp(-c_hat * lam[0])
    B_hat = denom1 / (x1 * (r - 1))
    A_hat = E1 - B_hat * x1
    theta = np.array([A_hat, B_hat, c_hat])
    logL = -np.sum(np.log(sigma)) - 1.5 * np.log(2 * np.pi)
    dE_dA = np.ones(3)
    dE_dB = np.exp(-c_hat * lam)
    dE_dc = -B_hat * lam * np.exp(-c_hat * lam)
    J = np.stack([dE_dA, dE_dB, dE_dc], axis=1)
    try:
        Jinv = np.linalg.inv(J)
        Sigma_post = Jinv @ np.diag(sigma ** 2) @ Jinv.T
    except np.linalg.LinAlgError:
        return None
    return theta, Sigma_post, logL


PRIOR_RANGES = {"linear": [(-3, 1), (-3, 3)], "quadratic": [(-3, 1), (-3, 3), (-3, 3)],
                 "exponential": [(-3, 1), (-3, 3), (1e-3, 8.0)]}


def prior_volume(model, Arange):
    ranges = [Arange] + PRIOR_RANGES[model][1:]
    v = 1.0
    for lo, hi in ranges:
        v *= (hi - lo)
    return v


def laplace_log_evidence(k, logL, Sigma_post, V_prior):
    sign, logdet = np.linalg.slogdet(Sigma_post)
    if sign <= 0:
        return -np.inf
    return logL + (k / 2) * np.log(2 * np.pi) + 0.5 * logdet - np.log(V_prior)


def bma_laplace_interval(counts, n_shots, coeffs, rng, Arange=(-3, 1), n_mix=6000):
    q_mle = counts / n_shots
    E_obs = np.array([E_of_q_group(q_mle[i], coeffs) for i in range(3)])
    var_q = q_mle * (1 - q_mle) / n_shots
    var_E = np.array([np.sum(coeffs ** 2 * 4 * var_q[i]) for i in range(3)])
    sigma_obs = np.sqrt(np.maximum(var_E, 1e-12))

    th_lin, Sig_lin, logL_lin = fit_linear_wls(E_obs, sigma_obs, lambdas)
    th_quad, Sig_quad, logL_quad = fit_quadratic_exact(E_obs, sigma_obs, lambdas)
    exp_fit = fit_exponential_exact(E_obs, sigma_obs, lambdas)

    logEv_lin = laplace_log_evidence(2, logL_lin, Sig_lin, prior_volume("linear", Arange))
    logEv_quad = laplace_log_evidence(3, logL_quad, Sig_quad, prior_volume("quadratic", Arange))
    if exp_fit is not None:
        th_exp, Sig_exp, logL_exp = exp_fit
        logEv_exp = laplace_log_evidence(3, logL_exp, Sig_exp, prior_volume("exponential", Arange))
    else:
        logEv_exp = -np.inf

    les = np.array([logEv_lin, logEv_quad, logEv_exp])
    finite = np.isfinite(les)
    if not finite.any():
        return None
    m = les[finite].max()
    w = np.zeros(3)
    w[finite] = np.exp(les[finite] - m)
    w /= w.sum()

    A_lin_s = rng.normal(th_lin[0], np.sqrt(Sig_lin[0, 0]), n_mix)
    A_quad_s = rng.normal(th_quad[0], np.sqrt(max(Sig_quad[0, 0], 1e-14)), n_mix)
    if exp_fit is not None:
        A_exp_s = rng.normal(th_exp[0], np.sqrt(max(Sig_exp[0, 0], 1e-14)), n_mix)
    else:
        A_exp_s = np.full(n_mix, np.nan)
    choice = rng.choice(['lin', 'quad', 'exp'], size=n_mix, p=w)
    A_mix = np.select([choice == 'lin', choice == 'quad', choice == 'exp'],
                       [A_lin_s, A_quad_s, A_exp_s])
    A_mix = A_mix[np.isfinite(A_mix)]
    if len(A_mix) < 50:
        return None
    return tuple(np.percentile(A_mix, [2.5, 97.5]))


# ---------------- Spike-and-slab ----------------
def spike_evidence(E_obs, sigma, lam, Arange, Srange):
    theta, Sigma_post, logL = fit_linear_wls(E_obs, sigma, lam)
    sign, logdet = np.linalg.slogdet(Sigma_post)
    V_prior = (Arange[1] - Arange[0]) * (Srange[1] - Srange[0])
    logEv = logL + np.log(2 * np.pi) + 0.5 * logdet - np.log(V_prior)
    return logEv, theta[0], Sigma_post


def gauss_logpdf_vec(E_obs, pred, sigma):
    resid = (E_obs[None, :] - pred) / sigma[None, :]
    return -0.5 * np.sum(resid ** 2, axis=-1) - np.sum(np.log(sigma)) - 0.5 * len(sigma) * np.log(2 * np.pi)


def slab_evidence_and_posterior(E_obs, sigma, lam, Arange, Srange, theta_scale, n_A=90, n_S=90, n_c=45, k_shape=3.0):
    A = np.linspace(*Arange, n_A)
    S = np.linspace(*Srange, n_S)
    c_max = theta_scale * (k_shape + 6 * np.sqrt(k_shape))
    c = np.linspace(1e-4, c_max, n_c)
    AA, SS, CC = np.meshgrid(A, S, c, indexing='ij')
    pred = AA[..., None] - (SS[..., None] / CC[..., None]) * (1 - np.exp(-CC[..., None] * lam[None, None, None, :]))
    logL = gauss_logpdf_vec(E_obs, pred, sigma)
    log_prior_c = gamma_dist.logpdf(CC, a=k_shape, scale=theta_scale)
    log_joint = logL + log_prior_c
    m = log_joint.max()
    cellA = (Arange[1] - Arange[0]) / (n_A - 1)
    cellS = (Srange[1] - Srange[0]) / (n_S - 1)
    cellc = c[1] - c[0]
    V_AS = (Arange[1] - Arange[0]) * (Srange[1] - Srange[0])
    logEv = m + np.log(np.sum(np.exp(log_joint - m)) * cellA * cellS * cellc) - np.log(V_AS)
    w = np.exp(log_joint - m)
    w /= w.sum()
    A_marg_coarse = w.sum(axis=(1, 2))
    A_marg_coarse /= A_marg_coarse.sum()
    A_fine = np.linspace(*Arange, 250)
    A_marg = np.interp(A_fine, A, A_marg_coarse)
    A_marg /= A_marg.sum()
    return logEv, A_fine, A_marg


def per_term_zscore(counts, n_shots):
    q = counts / n_shots
    q = np.clip(q, 1e-4, 1 - 1e-4)
    mid_pred = (q[0] + q[2]) / 2
    dev = q[1] - mid_pred
    var_q = q * (1 - q) / n_shots
    se = np.sqrt(var_q[1] + 0.25 * var_q[0] + 0.25 * var_q[2])
    return dev / np.maximum(se, 1e-10)


def calibrate_theta_scale(counts, n_shots):
    z = per_term_zscore(counts, n_shots)
    z = z[np.isfinite(z)]
    med = np.median(z)
    mad = np.median(np.abs(z - med))
    robust_std = 1.4826 * mad
    return np.sqrt(max(robust_std ** 2 - 1.0, 0.01))


def spike_slab_interval(counts, n_shots, coeffs, rng, Arange, theta_scale, Srange=(-3, 3), pi0=0.5):
    q_mle = counts / n_shots
    E_obs = np.array([E_of_q_group(q_mle[i], coeffs) for i in range(3)])
    var_q = q_mle * (1 - q_mle) / n_shots
    var_E = np.array([np.sum(coeffs ** 2 * 4 * var_q[i]) for i in range(3)])
    sigma_obs = np.sqrt(np.maximum(var_E, 1e-12))

    logEv_spike, Aprime_hat, Sig_lin = spike_evidence(E_obs, sigma_obs, lambdas, Arange, Srange)
    logEv_slab, A_grid, A_marg_slab = slab_evidence_and_posterior(E_obs, sigma_obs, lambdas, Arange, Srange, theta_scale)

    log_prior_odds = np.log(pi0) - np.log(1 - pi0)
    log_post_odds = log_prior_odds + logEv_spike - logEv_slab
    p_spike = 1 / (1 + np.exp(-log_post_odds))

    A_fine = np.linspace(*Arange, 250)
    A_marg_spike = norm.pdf(A_fine, loc=Aprime_hat, scale=np.sqrt(Sig_lin[0, 0]))
    A_marg_spike /= A_marg_spike.sum()
    A_marg_slab_i = np.interp(A_fine, A_grid, A_marg_slab)
    A_marg_slab_i /= A_marg_slab_i.sum()

    mix = p_spike * A_marg_spike + (1 - p_spike) * A_marg_slab_i
    mix /= mix.sum()
    cdf = np.cumsum(mix)
    lo_idx = np.searchsorted(cdf, 0.025)
    hi_idx = np.searchsorted(cdf, 0.975)
    return A_fine[lo_idx], A_fine[hi_idx], p_spike
