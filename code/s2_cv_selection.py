"""Step 2. Choose Beijing's control cities with leave-nv-out cross-validation, CV(nv).

Paper Section 3.1 (Shao 1993). For every subset gamma of the 21 candidate cities:
  1. draw nv of the T1 pre-policy months as a validation set (the rest, nc months, train);
  2. regress Beijing on a constant and the cities in gamma using the training months;
  3. record the mean squared prediction error on the validation months;
  4. repeat M times and average -> Gamma(gamma), eq. (10).
The subset with the smallest Gamma is the selected control group.

Settings follow the paper: nc = int(T1^0.75), nv = T1 - nc, M = 2*T1.
Subsets with more than nc - 1 cities are skipped: they cannot be estimated from nc
training months.

Speed: the M training/validation splits are fixed once; for each split we store the
cross-product matrices X'X, X'y of the training months and of the validation months.
A subset then only needs a small batched linear solve, no refitting from raw data.

Outputs (outputs/step2_cv/):
    table3_cv_top10.csv        ten smallest Gamma, plus the paper's 7-city group
    table4_best_by_size.csv    best subset for each number of control cities
    table5_seed_robustness.csv selected subset and paper-group rank under other random splits
    fig2_cv_by_size.png        Gamma of the best subset vs number of control cities
"""
import itertools
import multiprocessing as mp
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from s1_build_dataset import load_panel

warnings.filterwarnings("ignore")
OUT = C.OUT_DIR / "step2_cv"
CHUNK = 1024                      # subsets per batched solve
_S = {}                           # per-process copy of the split statistics


def make_splits(y, X, nv, M, seed=42):
    """Fix M random validation sets and precompute their cross-products."""
    T1 = len(y)
    Xf = np.column_stack([np.ones(T1), X])
    rng = np.random.RandomState(seed)
    G, b, H, c, s = [], [], [], [], []
    for _ in range(M):
        test = rng.choice(T1, size=nv, replace=False)
        train = np.setdiff1d(np.arange(T1), test)
        G.append(Xf[train].T @ Xf[train]); b.append(Xf[train].T @ y[train])
        H.append(Xf[test].T @ Xf[test]);   c.append(Xf[test].T @ y[test])
        s.append(y[test] @ y[test])
    return dict(G=np.array(G), b=np.array(b), H=np.array(H),
                c=np.array(c), s=np.array(s), nv=nv)


def _init(stats):
    _S.update(stats)


def gamma_chunk(subsets):
    """Gamma for a batch of equally sized subsets. subsets: (n, k) city indices."""
    n = len(subsets)
    I = np.hstack([np.zeros((n, 1), dtype=int), subsets + 1])        # column 0 = constant
    G, b, H, c, s = (_S[k] for k in ("G", "b", "H", "c", "s"))
    Gs = G[:, I[:, :, None], I[:, None, :]]                          # (M, n, k+1, k+1)
    bs, cs, Hs = b[:, I], c[:, I], H[:, I[:, :, None], I[:, None, :]]
    try:
        beta = np.linalg.solve(Gs, bs[..., None])[..., 0]            # (M, n, k+1)
    except np.linalg.LinAlgError:
        return np.full(n, np.inf)
    err = s[:, None] - 2 * (beta * cs).sum(-1) \
        + (beta * (Hs @ beta[..., None])[..., 0]).sum(-1)
    return err.mean(axis=0) / _S["nv"]


def run_cv(y, X, nv, M, max_size, workers, seed=42):
    N = X.shape[1]
    stats = make_splits(y, X, nv, M, seed)
    jobs, meta = [], []
    for k in range(1, max_size + 1):
        subs = np.array(list(itertools.combinations(range(N), k)))
        for i in range(0, len(subs), CHUNK):
            jobs.append(subs[i:i + CHUNK])
    total = sum(len(j) for j in jobs)
    print(f"  {total:,} subsets (sizes 1..{max_size}), {workers} worker processes")
    t0, gammas = time.time(), []
    with mp.Pool(workers, initializer=_init, initargs=(stats,)) as pool:
        for i, g in enumerate(pool.imap(gamma_chunk, jobs), 1):
            gammas.append(g)
            if i % 100 == 0 or i == len(jobs):
                print(f"\r  {i}/{len(jobs)} batches, {time.time() - t0:.0f}s", end="", flush=True)
    print()
    subsets = np.vstack([np.hstack([j, np.full((len(j), max_size - j.shape[1]), -1)]) for j in jobs])
    return np.concatenate(gammas), subsets


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre = load_panel().loc[C.CASES["beijing"]["pre"][0]:C.CASES["beijing"]["pre"][1]]
    pool_cities = C.BEIJING_POOL
    y, X = pre["Beijing"].values, pre[pool_cities].values
    T1 = len(y)
    nc = int(T1 ** 0.75)
    nv, M = T1 - nc, 2 * T1
    print(f"T1={T1}, nc={nc}, nv={nv}, M={M}, candidate cities={len(pool_cities)}")

    gamma, subsets = run_cv(y, X, nv, M, max_size=nc - 1, workers=max(1, mp.cpu_count() - 1))
    names = lambda row: [pool_cities[i] for i in row if i >= 0]
    df = pd.DataFrame({"gamma": gamma, "cities": [", ".join(sorted(names(r))) for r in subsets],
                       "n_cities": (subsets >= 0).sum(1)})
    df = df.sort_values("gamma").reset_index(drop=True)
    df["rank"] = df.index + 1

    paper = ", ".join(sorted(C.CASES["beijing"]["controls"]))
    top = df.head(10)[["rank", "gamma", "n_cities", "cities"]].copy()
    top["note"] = ""
    paper_row = df[df["cities"] == paper][["rank", "gamma", "n_cities", "cities"]].copy()
    paper_row["note"] = "paper's selection"
    pd.concat([top, paper_row]).drop_duplicates("rank").to_csv(
        OUT / "table3_cv_top10.csv", index=False, float_format="%.4f")

    by_size = df.loc[df.groupby("n_cities")["gamma"].idxmin()].sort_values("n_cities")
    by_size[["n_cities", "gamma", "cities"]].to_csv(
        OUT / "table4_best_by_size.csv", index=False, float_format="%.4f")

    # Figure 2: Gamma of the best subset for each size
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(by_size["n_cities"], by_size["gamma"], "-o", color=C.BLUE, lw=2, ms=5)
    best = df.iloc[0]
    ax.scatter(best["n_cities"], best["gamma"], s=90, color=C.ORANGE, zorder=3,
               label=f"CV(nv) minimum: {int(best['n_cities'])} cities")
    if len(paper_row):
        p = paper_row.iloc[0]
        ax.scatter(p["n_cities"], p["gamma"], s=90, facecolor="none", edgecolor="black",
                   lw=1.5, zorder=3, label=f"Paper's 7 cities (rank {int(p['rank']):,})")
    ax.set_xlabel("Number of control cities")
    ax.set_ylabel(r"CV($n_v$) prediction error $\Gamma$")
    ax.set_title("Best subset at each size: more controls stop helping", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(OUT / "fig2_cv_by_size.png", dpi=250)
    plt.close(fig)

    # Robustness: CV uses random splits, so repeat with other seeds.
    rows = []
    for seed in (1, 2, 3, 4):
        g, sub = run_cv(y, X, nv, M, nc - 1, max(1, mp.cpu_count() - 1), seed)
        d = pd.DataFrame({"gamma": g, "cities": [", ".join(sorted(names(r))) for r in sub]})
        d = d.sort_values("gamma").reset_index(drop=True)
        hit = d.index[d["cities"] == paper]
        rows.append({"seed": seed, "best_cities": d.loc[0, "cities"], "best_gamma": d.loc[0, "gamma"],
                     "paper_group_rank": int(hit[0]) + 1 if len(hit) else None})
    rows.insert(0, {"seed": 42, "best_cities": df.loc[0, "cities"], "best_gamma": df.loc[0, "gamma"],
                    "paper_group_rank": int(paper_row.iloc[0]["rank"]) if len(paper_row) else None})
    pd.DataFrame(rows).to_csv(OUT / "table5_seed_robustness.csv", index=False, float_format="%.4f")

    print("\nTop 10 subsets by Gamma:")
    print(top[["rank", "gamma", "n_cities", "cities"]].to_string(index=False))
    if len(paper_row):
        print(f"\nPaper's 7-city group: rank {int(paper_row.iloc[0]['rank']):,}, "
              f"Gamma {paper_row.iloc[0]['gamma']:.4f}")
    print("\nOther random splits:\n", pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
