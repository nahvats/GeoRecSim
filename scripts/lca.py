"""Latent class analysis with mixed indicators (Vermunt & Magidson 2016).

Used by sim_results_analysis.ipynb to build recommender-response "personas"
in the style of Ng, Mishra & Chen (2025, J. Transport & Health 44:102135).

Model (their Eq. 1):  f(y_i) = sum_k P(k) prod_h f(y_ih | k)
  * continuous indicators: class-specific Gaussian (mean, variance)
  * binary indicators:     class-specific Bernoulli
  * indicators are conditionally independent given the class
  * missing indicator values are skipped in the product (FIML under MAR)

Estimated by EM from several random starts; the best log-likelihood is kept and
`replicated` reports whether the best LL was reached by more than one start
(the paper's replication check for local maxima).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class LCAResult:
    k: int
    loglik: float
    n_params: int
    n: int
    weights: np.ndarray          # (k,)
    means: np.ndarray            # (k, n_cont)
    variances: np.ndarray        # (k, n_cont)
    probs: np.ndarray            # (k, n_bin)
    posterior: np.ndarray        # (n, k)
    replicated: bool

    @property
    def aic(self):
        return -2 * self.loglik + 2 * self.n_params

    @property
    def bic(self):
        return -2 * self.loglik + np.log(self.n) * self.n_params

    @property
    def sabic(self):
        return -2 * self.loglik + np.log((self.n + 2) / 24) * self.n_params

    @property
    def entropy(self):
        """Relative entropy in [0, 1]; 1 = perfectly separated classes."""
        if self.k == 1:
            return np.nan
        p = np.clip(self.posterior, 1e-12, 1)
        return 1 - (-(p * np.log(p)).sum()) / (self.n * np.log(self.k))

    @property
    def modal(self):
        return self.posterior.argmax(1)


def _loglik_parts(Xc, Xb, w, mu, var, pr):
    """(n, k) log of P(k) * prod_h f(y_ih | k), skipping NaNs."""
    n, k = Xc.shape[0], len(w)
    L = np.tile(np.log(w), (n, 1))
    if Xc.shape[1]:
        obs = ~np.isnan(Xc)
        x = np.where(obs, Xc, 0.0)
        for j in range(k):
            ll = -0.5 * (np.log(2 * np.pi * var[j]) + (x - mu[j]) ** 2 / var[j])
            L[:, j] += np.where(obs, ll, 0.0).sum(1)
    if Xb.shape[1]:
        obs = ~np.isnan(Xb)
        x = np.where(obs, Xb, 0.0)
        p = np.clip(pr, 1e-9, 1 - 1e-9)
        for j in range(k):
            ll = x * np.log(p[j]) + (1 - x) * np.log(1 - p[j])
            L[:, j] += np.where(obs, ll, 0.0).sum(1)
    return L


def _em_once(Xc, Xb, k, rng, max_iter, tol, var_floor):
    n = Xc.shape[0]
    # random soft start: Dirichlet responsibilities
    R = rng.dirichlet(np.ones(k), size=n)
    prev = -np.inf
    for _ in range(max_iter):
        # M step
        Nk = R.sum(0) + 1e-12
        w = Nk / n
        oc, ob = ~np.isnan(Xc), ~np.isnan(Xb)
        xc, xb = np.nan_to_num(Xc), np.nan_to_num(Xb)
        mu = np.stack([(R[:, [j]] * xc).sum(0) / (R[:, [j]] * oc).sum(0).clip(1e-12) for j in range(k)])
        var = np.stack([(R[:, [j]] * oc * (xc - mu[j]) ** 2).sum(0) / (R[:, [j]] * oc).sum(0).clip(1e-12)
                        for j in range(k)]).clip(var_floor)
        pr = np.stack([(R[:, [j]] * xb).sum(0) / (R[:, [j]] * ob).sum(0).clip(1e-12) for j in range(k)])
        # E step
        L = _loglik_parts(Xc, Xb, w, mu, var, pr)
        tot = logsumexp(L, axis=1)
        ll = tot.sum()
        R = np.exp(L - tot[:, None])
        if abs(ll - prev) < tol * abs(ll):
            break
        prev = ll
    return ll, w, mu, var, pr, R


def fit_lca(Xc: np.ndarray, Xb: np.ndarray, k: int, n_starts: int = 10, seed: int = 0,
            max_iter: int = 500, tol: float = 1e-8, var_floor: float = 1e-4) -> LCAResult:
    rng = np.random.default_rng(seed)
    best, lls = None, []
    for _ in range(n_starts):
        out = _em_once(Xc, Xb, k, rng, max_iter, tol, var_floor)
        lls.append(out[0])
        if best is None or out[0] > best[0]:
            best = out
    ll, w, mu, var, pr, R = best
    # canonical order: largest class first
    order = np.argsort(-w)
    w, mu, var, pr, R = w[order], mu[order], var[order], pr[order], R[:, order]
    n_params = (k - 1) + k * (2 * Xc.shape[1] + Xb.shape[1])
    replicated = sum(abs(l - ll) < 1e-6 * abs(ll) + 1e-3 for l in lls) > 1
    return LCAResult(k, ll, n_params, Xc.shape[0], w, mu, var, pr, R, replicated)
