"""Step 7. Monte Carlo comparison of CV(nv), AICC and AIC for choosing control cities.

Replicates paper Section 3.2 / Table 1. The setup follows the paper:
  - N = 21 cities: one treated city and 20 candidate controls; T1 in {25, 40, 60}
    pre-policy periods and T2 = 10 post-policy periods;
  - 1, 2 and 3 common factors with the AR/MA processes given in the paper;
  - factor loadings b_i ~ N(1, 1), errors eps ~ N(0, sigma2), sigma2 in {1, 0.5, 0.1};
  - each method picks control cities from the pre-policy data; we then predict the
    post-policy outcome of the treated city and record the prediction error (PMSE).

Choices the paper leaves open (our reading):
  - CV(nv): nc = int(T1^0.75), nv = T1 - nc, M = 2*T1 random splits, as in step 2.
  - AIC / AICC (Hsiao, Ching & Wan 2012): for each number m of controls take the subset with
    the smallest residual sum of squares (highest R2), then choose m by AIC or AICC.
    AIC = T1*ln(RSS/T1) + 2k, AICC = AIC + 2k(k+1)/(T1-k-1), with k = m + 1 coefficients.
  - The PMSE compares the prediction with the realised outcome, error term included.
  - The factor processes start at 0 and run 100 periods before the sample (burn-in).
  - Evidence that sigma2 is the error *variance*: PMSE in Table 1 scales roughly like sigma2.

Every method searches ALL 2^20 - 1 subsets, so one replication is slow (minutes). Each
replication is an independent task spread over the CPU cores, and finished tasks are saved
to raw_results.csv, so an interrupted run can simply be started again.

Usage (from code/):
    python s7_monte_carlo.py --reps 20            # full grid, a few hours
    python s7_monte_carlo.py --reps 1 --t1 25 --factors 1 --sigma2 1   # quick test
    python s7_monte_carlo.py --summary-only       # rebuild tables and figure

Outputs (outputs/step7_monte_carlo/):
    raw_results.csv                  one row per replication
    table14_monte_carlo.csv          our averages next to paper Table 1
    fig10_cv_vs_aic.png              PMSE ratios AICC/CV and AIC/CV, ours vs paper
"""
import os

# One thread per process: the parallelism is across replications.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import multiprocessing as mp
import time
import warnings
from functools import lru_cache

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from paper_values import PAPER_TABLE1
from s2_cv_selection import make_splits

warnings.filterwarnings("ignore")
OUT = C.OUT_DIR / "step7_monte_carlo"
RAW = OUT / "raw_results.csv"

N_CONTROLS, T2, BURN_IN, BASE_SEED = 20, 10, 100, 6013
METHODS = ("CV", "AICC", "AIC")
# (AR coefficient, MA coefficients) of each factor, from the paper
FACTOR_SPECS = {
    1: [(0.95, [])],
    2: [(0.3, []), (0.6, [])],
    3: [(0.8, []), (-0.6, [0.8]), (0.0, [0.9, 0.4])],
}


def simulate(rng, n_factors, sigma2, T1):
    """Outcome of the treated city (T1 + T2 periods) and of the 20 candidate controls."""
    n = T1 + T2 + BURN_IN
    u = rng.standard_normal((n + 2, n_factors))          # two extra rows for the MA lags
    f = np.zeros((n, n_factors))
    for j, (ar, ma) in enumerate(FACTOR_SPECS[n_factors]):
        for t in range(n):
            f[t, j] = (ar * f[t - 1, j] if t else 0.0) + u[t + 2, j] \
                      + sum(m * u[t + 2 - lag, j] for lag, m in enumerate(ma, 1))
    b = rng.normal(1.0, 1.0, size=(N_CONTROLS + 1, n_factors))
    g = f[BURN_IN:] @ b.T + rng.normal(0.0, np.sqrt(sigma2), size=(T1 + T2, N_CONTROLS + 1))
    return g[:, 0], g[:, 1:]


@lru_cache(maxsize=None)
def combos(k):
    """All subsets of k of the 20 controls, as an int8 array."""
    return np.array(list(itertools.combinations(range(N_CONTROLS), k)), dtype=np.int8)


def score_subsets(I, full, splits, with_cv):
    """Residual sum of squares on the full pre-period and, optionally, CV Gamma.

    I: (n, k+1) column indices into [1, X] (0 is the constant), one row per subset.
    """
    G0, b0, yy = full
    G = G0[I[:, :, None], I[:, None, :]]
    b = b0[I]
    beta = np.linalg.solve(G, b[..., None])[..., 0]
    rss = yy - (beta * b).sum(-1)
    if not with_cv:
        return rss, None
    Gs = splits["G"][:, I[:, :, None], I[:, None, :]]          # (M, n, k+1, k+1)
    Hs = splits["H"][:, I[:, :, None], I[:, None, :]]
    bs, cs = splits["b"][:, I], splits["c"][:, I]
    beta = np.linalg.solve(Gs, bs[..., None])[..., 0]
    err = splits["s"][:, None] - 2 * (beta * cs).sum(-1) \
        + (beta * (Hs @ beta[..., None])[..., 0]).sum(-1)
    return rss, err.mean(0) / splits["nv"]


def search(y, X, rng):
    """Control cities chosen by CV(nv), AICC and AIC from the pre-policy sample."""
    T1 = len(y)
    nc = int(T1 ** 0.75)
    nv, M = T1 - nc, 2 * T1
    D = np.column_stack([np.ones(T1), X])
    full = (D.T @ D, D.T @ y, y @ y)
    splits = make_splits(y, X, nv, M, seed=int(rng.integers(2 ** 31 - 1)))

    best_rss = {}                      # number of controls -> (smallest RSS, subset)
    best_cv = (np.inf, None)           # (smallest Gamma, subset)
    for k in range(1, N_CONTROLS + 1):
        subsets = combos(k)
        with_cv = k <= nc - 1          # more controls cannot be fitted on nc training months
        step = int(min(4096, max(16, 3e6 // ((k + 1) ** 2 * (M if with_cv else 1)))))
        best_rss[k] = (np.inf, None)
        for i in range(0, len(subsets), step):
            part = subsets[i:i + step].astype(np.intp)
            I = np.hstack([np.zeros((len(part), 1), dtype=np.intp), part + 1])
            try:
                rss, gamma = score_subsets(I, full, splits, with_cv)
            except np.linalg.LinAlgError:
                continue
            j = int(np.nanargmin(rss))
            if rss[j] < best_rss[k][0]:
                best_rss[k] = (rss[j], part[j])
            if with_cv:
                j = int(np.nanargmin(gamma))
                if gamma[j] < best_cv[0]:
                    best_cv = (gamma[j], part[j])

    ks = np.arange(1, N_CONTROLS + 1)
    rss = np.maximum([best_rss[k][0] for k in ks], 1e-12)
    p = ks + 1                                           # coefficients incl. constant
    aic = T1 * np.log(rss / T1) + 2 * p
    aicc = aic + 2 * p * (p + 1) / (T1 - p - 1)
    return {"CV": best_cv[1],
            "AICC": best_rss[int(ks[np.argmin(aicc)])][1],
            "AIC": best_rss[int(ks[np.argmin(aic)])][1]}


def post_pmse(y, X, T1, subset):
    """Fit on the pre-period with the chosen controls, mean squared error after it."""
    D = np.column_stack([np.ones(T1), X[:T1][:, subset]])
    coef = np.linalg.lstsq(D, y[:T1], rcond=None)[0]
    pred = np.column_stack([np.ones(len(y) - T1), X[T1:][:, subset]]) @ coef
    return float(np.mean((y[T1:] - pred) ** 2))


def run_task(task):
    n_factors, sigma2, T1, rep = task
    t0 = time.time()
    rng = np.random.default_rng(np.random.SeedSequence(
        [BASE_SEED, n_factors, int(round(sigma2 * 10)), T1, rep]))
    y, X = simulate(rng, n_factors, sigma2, T1)
    chosen = search(y[:T1], X[:T1], rng)
    row = {"n_factors": n_factors, "sigma2": sigma2, "T1": T1, "rep": rep}
    for m in METHODS:
        row[f"{m}_n"] = len(chosen[m])
        row[f"{m}_pmse"] = post_pmse(y, X, T1, chosen[m])
        row[f"{m}_cities"] = " ".join(str(int(c)) for c in sorted(chosen[m]))
    row["seconds"] = round(time.time() - t0, 1)
    return row


def summarize():
    raw = pd.read_csv(RAW)
    keys = ["n_factors", "T1", "sigma2"]
    rows = []
    for (nf, t1, s2), d in raw.groupby(keys):
        r = {"n_factors": nf, "T1": t1, "sigma2": s2, "reps": len(d)}
        for m in METHODS:
            r[f"{m}_n"], r[f"{m}_pmse"] = d[f"{m}_n"].mean(), d[f"{m}_pmse"].mean()
            r[f"paper_{m}_n"], r[f"paper_{m}_pmse"] = PAPER_TABLE1[(nf, t1, s2, m)]
        r["share_CV_beats_AICC"] = (d["CV_pmse"] < d["AICC_pmse"]).mean()
        r["share_CV_beats_AIC"] = (d["CV_pmse"] < d["AIC_pmse"]).mean()
        rows.append(r)
    tab = pd.DataFrame(rows).sort_values(["n_factors", "T1", "sigma2"], ascending=[True, True, False])
    tab = tab.reset_index(drop=True)
    tab.to_csv(OUT / "table14_monte_carlo.csv", index=False, float_format="%.4f")

    ours_best = (tab["CV_pmse"] <= tab[["AICC_pmse", "AIC_pmse"]].min(axis=1)).sum()
    paper_best = (tab["paper_CV_pmse"] <= tab[["paper_AICC_pmse", "paper_AIC_pmse"]].min(axis=1)).sum()
    print(tab[["n_factors", "T1", "sigma2", "reps", "CV_pmse", "AICC_pmse", "AIC_pmse",
               "CV_n", "AICC_n", "AIC_n"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nCV(nv) has the smallest PMSE in {ours_best} of {len(tab)} cells "
          f"(paper: {paper_best} of {len(tab)}).")

    # Figure 10: PMSE of the alternative / PMSE of CV(nv); above 1 means CV(nv) is better.
    x = np.arange(len(tab))
    labels = [f"{r.n_factors}F, T1={r.T1}, s2={r.sigma2:g}" for r in tab.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for ax, m in zip(axes, ("AICC", "AIC")):
        ax.axhline(1, color=C.GREY, lw=1)
        ax.scatter(x, tab[f"{m}_pmse"] / tab["CV_pmse"], color=C.BLUE, s=28, label="Our simulation")
        ax.scatter(x, tab[f"paper_{m}_pmse"] / tab["paper_CV_pmse"], facecolor="none",
                   edgecolor=C.ORANGE, s=42, lw=1.4, label="Paper Table 1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_title(f"PMSE({m}) / PMSE(CV(nv)); above 1 = CV(nv) better", fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "fig10_cv_vs_aic.png", dpi=250)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--t1", type=int, nargs="+", default=[25, 40, 60])
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--sigma2", type=float, nargs="+", default=[1.0, 0.5, 0.1])
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if not a.summary_only:
        done = set()
        if RAW.exists():
            old = pd.read_csv(RAW)
            done = set(zip(old["n_factors"], old["sigma2"], old["T1"], old["rep"]))
        tasks = [(f, s, t, r) for t in sorted(a.t1, reverse=True) for f in a.factors
                 for s in a.sigma2 for r in range(a.reps) if (f, s, t, r) not in done]
        print(f"{len(tasks)} replications to run ({len(done)} already saved), {a.workers} processes")
        t0 = time.time()
        with mp.Pool(a.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(run_task, tasks), 1):
                pd.DataFrame([row]).to_csv(RAW, mode="a", header=not RAW.exists(), index=False)
                if i % 5 == 0 or i == len(tasks):
                    el = time.time() - t0
                    print(f"  {i}/{len(tasks)} done, {el / 60:.0f} min elapsed, "
                          f"about {el / i * (len(tasks) - i) / 60:.0f} min left", flush=True)
    summarize()


if __name__ == "__main__":
    main()
