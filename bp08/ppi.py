from __future__ import annotations

import numpy as np
from scipy.stats import norm


def krippendorff_alpha_nominal(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    idx = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        o[idx[x], idx[y]] += 1
        o[idx[y], idx[x]] += 1
    n_c = o.sum(1)
    n = n_c.sum()
    disagree_obs = o.sum() - np.trace(o)
    disagree_exp = (n_c.sum() ** 2 - (n_c ** 2).sum()) / (n - 1)
    return 1.0 if disagree_exp == 0 else float(1 - disagree_obs / disagree_exp)


def wilson(k, n, z):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def estimate(Y, f, f_hard, sample, level=0.95) -> dict:
    N, n = len(Y), len(sample)
    z = norm.ppf(0.5 + level / 2)
    mask = np.zeros(N, bool)
    mask[sample] = True
    yS, fS, fU = Y[mask], f[mask], f[~mask]
    w = (N - n) / N
    out = {}
    lo, hi = wilson(yS.sum(), n, z)
    fpc = np.sqrt((N - n) / (N - 1))
    m = yS.mean()
    out["human_only"] = (m, m - (m - lo) * fpc, m + (hi - m) * fpc)
    mh = f_hard.mean()
    se = np.sqrt(mh * (1 - mh) / N)
    out["naive"] = (mh, mh - z * se, mh + z * se)
    th_u = fU.mean() + (yS - fS).mean()
    se_u = np.sqrt(fU.var(ddof=1) / len(fU) + (yS - fS).var(ddof=1) / n)
    th = (1 - w) * m + w * th_u
    out["ppi"] = (th, th - z * w * se_u, th + z * w * se_u)
    vf = f.var(ddof=1)
    lam = 0.0 if vf == 0 else float(np.clip(np.cov(yS, fS, ddof=1)[0, 1] / ((1 + n / (N - n)) * vf), 0, 1))
    th_u = yS.mean() + lam * (fU.mean() - fS.mean())
    se_u = np.sqrt(lam ** 2 * fU.var(ddof=1) / len(fU) + (yS - lam * fS).var(ddof=1) / n)
    th = (1 - w) * m + w * th_u
    out["ppi++"] = (th, th - z * w * se_u, th + z * w * se_u)
    return out


def simulate(Y, f, f_hard, n, reps, rng, level=0.95) -> dict:
    truth = Y.mean()
    acc = {k: [] for k in ("human_only", "naive", "ppi", "ppi++")}
    for _ in range(reps):
        S = rng.choice(len(Y), n, replace=False)
        for k, (th, lo, hi) in estimate(Y, f, f_hard, S, level).items():
            acc[k].append((lo <= truth <= hi, hi - lo, th - truth))
    return {k: {"coverage": float(np.mean([a for a, _, _ in v])), "ci_width": float(np.mean([b for _, b, _ in v])),
                "bias": float(np.mean([c for _, _, c in v])), "rmse": float(np.sqrt(np.mean([c * c for _, _, c in v])))}
            for k, v in acc.items()}
