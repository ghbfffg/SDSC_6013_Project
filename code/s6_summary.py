"""Step 6. Summary of the three policy evaluations against the paper.

Outputs (outputs/step6_summary/):
    table13_replication_summary.csv   our estimates vs paper Tables 3, 5, 7

Significance: "result" uses a two-sided test. The paper also calls Chongqing
(t = -1.60) significant at 10%; that holds only for a one-sided test (critical value
1.28), so both p-values are reported.
    fig9_three_cases.png              the three counterfactual paths side by side
"""
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

import config as C
from counterfactual import estimate
from paper_values import PAPER

OUT = C.OUT_DIR / "step6_summary"


def verdict(t):
    return "significant at 1%" if abs(t) > 2.576 else \
           "significant at 5%" if abs(t) > 1.96 else \
           "significant at 10%" if abs(t) > 1.645 else "not significant"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, results = [], {}
    for case in C.CASES:
        r = results[case] = estimate(case)
        p = PAPER[case]
        rows.append({
            "case": r["cfg"]["title"],
            "post_period": f"{C.CASES[case]['post'][0]} ~ {C.CASES[case]['post'][1]}",
            "n_controls": len(r["controls"]),
            "avg_actual": r["post"][r["cfg"]["target"]].mean(), "paper_avg_actual": p["avg_actual"],
            "avg_counterfactual": r["pred"]["mean"].mean(), "paper_avg_counterfactual": p["avg_cf"],
            "avg_effect": r["avg_effect"], "paper_avg_effect": p["avg_effect"],
            "t_stat": r["t"], "paper_t_stat": p["t"],
            "p_two_sided": 2 * stats.norm.sf(abs(r["t"])),
            "p_one_sided": stats.norm.sf(abs(r["t"])),
            "result": verdict(r["t"]) + " (two-sided)",
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table13_replication_summary.csv", index=False, float_format="%.4f")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for ax, (case, r) in zip(axes, results.items()):
        post, y, p = r["post"], r["cfg"]["target"], r["pred"]
        x = range(len(post))
        ax.fill_between(x, p["mean_ci_lower"], p["mean_ci_upper"], color=C.ORANGE, alpha=0.18, lw=0)
        ax.plot(x, post[y], color=C.BLUE, lw=2, label="Actual")
        ax.plot(x, p["mean"], color=C.ORANGE, lw=2, ls="--", label="Counterfactual")
        ax.set_xticks(list(x)[::4])
        ax.set_xticklabels(post.index[::4], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{y}: {r['avg_effect']:+.2f} pp, t = {r['t']:.2f}", fontsize=9, loc="left")
        ax.grid(axis="y", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Year-on-year price growth (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "fig9_three_cases.png", dpi=250)
    plt.close(fig)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
