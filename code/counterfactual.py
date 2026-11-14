"""Counterfactual estimation shared by the three case scripts (s3, s4, s5).

Method (paper Section 3.1, steps 2-3):
  1. OLS of the treated city's growth rate on a constant and the control cities,
     using the pre-policy months.
  2. Predict the post-policy months from the fitted coefficients -> counterfactual.
  3. Effect = actual - counterfactual; the average effect is the policy estimate.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from paper_values import PAPER
from s1_build_dataset import load_panel

# The paper takes its t-statistic from Li & Bell (2017, Thm 3.2): estimation error of
# the pre-period regression plus the long-run variance of the monthly effects.
# Neither paper fixes the lag lengths. We use standard rules, which give 3 and 2 in all
# three cases and reproduce the t-values in Tables 3, 5, 7 of the paper:
#   regression covariance: Newey-West rule int(4 * (T1/100)^(2/9))
#   effect series:         round(T2^(1/4))


def long_run_var(e, lags):
    """Bartlett-kernel long-run variance of a series."""
    e = np.asarray(e) - np.mean(e)
    n = len(e)
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return v


def estimate(case, controls=None):
    """Fit on the pre-period and build the counterfactual. Returns a dict of results."""
    cfg = C.CASES[case]
    controls = controls or cfg["controls"]
    panel = load_panel()
    pre = panel.loc[cfg["pre"][0]:cfg["pre"][1]]
    post = panel.loc[cfg["post"][0]:cfg["post"][1]]
    y = cfg["target"]

    model = sm.OLS(pre[y], sm.add_constant(pre[controls])).fit()
    Xpost = sm.add_constant(post[controls], has_constant="add")
    pred = model.get_prediction(Xpost).summary_frame(alpha=0.05)
    effect = post[y] - pred["mean"]

    # t-statistic of the average effect
    xbar = Xpost.mean().values
    lag_beta = int(4 * (len(pre) / 100) ** (2 / 9))
    lag_effect = int(round(len(post) ** 0.25))
    cov = np.asarray(model.get_robustcov_results("HAC", maxlags=lag_beta).cov_params())
    var_param = xbar @ cov @ xbar
    var_effect = long_run_var(effect, lag_effect) / len(effect)
    se = np.sqrt(var_param + var_effect)
    return dict(case=case, cfg=cfg, controls=controls, pre=pre, post=post, model=model,
                pred=pred, effect=effect, avg_effect=effect.mean(), se=se,
                t=effect.mean() / se)


def monthly_table(r):
    p, y = r["pred"], r["cfg"]["target"]
    out = pd.DataFrame({
        "month": r["post"].index,
        "actual": r["post"][y].values,
        "counterfactual": p["mean"].values,
        "effect": r["effect"].values,
        "cf_ci_low": p["mean_ci_lower"].values,
        "cf_ci_high": p["mean_ci_upper"].values,
    })
    out["significant_5pct"] = (out["actual"] < out["cf_ci_low"]) | (out["actual"] > out["cf_ci_high"])
    if r["controls"] == r["cfg"]["controls"]:
        out["paper_effect"] = PAPER[r["case"]]["effect"]
    avg = {"month": "Average", "actual": out["actual"].mean(),
           "counterfactual": out["counterfactual"].mean(), "effect": r["avg_effect"]}
    if "paper_effect" in out:
        avg["paper_effect"] = PAPER[r["case"]]["avg_effect"]
    out = pd.concat([out, pd.DataFrame([avg])], ignore_index=True)
    out.loc[out.index[-1], "t_stat"] = r["t"]
    return out


def coefficient_table(r):
    m = r["model"]
    out = pd.DataFrame({"variable": m.params.index, "coef": m.params.values,
                        "std_error": m.bse.values, "t_stat": m.tvalues.values})
    paper = PAPER[r["case"]]["coef"]
    out["paper_coef"] = [paper[v][0] if v in paper else np.nan for v in out["variable"]]
    out["paper_std_error"] = [paper[v][1] if v in paper else np.nan for v in out["variable"]]
    out["paper_t_stat"] = [paper[v][2] if v in paper else np.nan for v in out["variable"]]
    r2 = pd.DataFrame([{"variable": "R2", "coef": m.rsquared, "paper_coef": PAPER[r["case"]]["r2"]}])
    return pd.concat([out, r2], ignore_index=True)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("Year-on-year price growth (%)")


def plot_pre_fit(r, path, label):
    """Paper Figs 3, 5, 7: actual vs fitted in the pre-policy period."""
    y, pre = r["cfg"]["target"], r["pre"]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    x = range(len(pre))
    ax.plot(x, pre[y], color=C.BLUE, lw=2, label=f"{y} actual")
    ax.plot(x, r["model"].fittedvalues, color=C.ORANGE, lw=2, ls="--", label="Fitted from control cities")
    ax.set_xticks(list(x)[::4])
    ax.set_xticklabels(pre.index[::4], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{label}: pre-policy fit ({pre.index[0]} to {pre.index[-1]}), "
                 f"R$^2$ = {r['model'].rsquared:.3f}", fontsize=10, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_counterfactual(r, path, label):
    """Paper Figs 4, 6, 8: actual vs counterfactual after the policy, with 95% band."""
    y, post, p = r["cfg"]["target"], r["post"], r["pred"]
    x = np.arange(len(post))
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    ax.fill_between(x, p["mean_ci_lower"], p["mean_ci_upper"], color=C.ORANGE, alpha=0.18,
                    lw=0, label="95% band of counterfactual")
    ax.plot(x, post[y], color=C.BLUE, lw=2, label=f"{y} actual")
    ax.plot(x, p["mean"], color=C.ORANGE, lw=2, ls="--", label="Counterfactual (no policy)")
    ax.set_xticks(x[::2])
    ax.set_xticklabels(post.index[::2], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{label}: average effect {r['avg_effect']:+.2f} pp (t = {r['t']:.2f})",
                 fontsize=10, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def run_case(case, out_dir, names):
    """Produce all tables and figures for one case. `names` = dict of file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    r = estimate(case)
    label = r["cfg"]["title"]
    coefficient_table(r).to_csv(out_dir / names["coef"], index=False, float_format="%.4f")
    monthly_table(r).to_csv(out_dir / names["effects"], index=False, float_format="%.4f")
    plot_pre_fit(r, out_dir / names["fig_fit"], label)
    plot_counterfactual(r, out_dir / names["fig_cf"], label)

    paper = PAPER[case]
    sig = monthly_table(r)["significant_5pct"].iloc[:-1].sum()
    print(f"{label}")
    print(f"  pre-period R2      : {r['model'].rsquared:.4f}   (paper {paper['r2']})")
    print(f"  average actual     : {r['post'][r['cfg']['target']].mean():.4f}   (paper {paper['avg_actual']})")
    print(f"  average counterf.  : {r['pred']['mean'].mean():.4f}   (paper {paper['avg_cf']})")
    print(f"  average effect     : {r['avg_effect']:.4f}   (paper {paper['avg_effect']})")
    print(f"  t-statistic        : {r['t']:.4f}   (paper {paper['t']})")
    print(f"  months outside 95% band: {sig} of {len(r['post'])}")
    return r
