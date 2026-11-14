"""Step 3. Beijing home-purchase restriction (policy start 2010-05), paper Section 4.1.

Main result uses the paper's 7 control cities (its Table 2). For comparison we also
run the control group that our own CV(nv) search selected in step 2 (rank 1).

Outputs (outputs/step3_beijing/):
    table6_coefficients.csv     OLS weights vs paper Table 2
    table7_effects.csv          monthly effects vs paper Table 3
    table8_control_groups.csv   paper's controls vs our CV-selected controls
    fig3_pre_fit.png            paper Fig 3
    fig4_counterfactual.png     paper Fig 4
    fig4b_control_groups.png    both counterfactuals on one plot
"""
import matplotlib.pyplot as plt
import pandas as pd

import config as C
from counterfactual import estimate, run_case, _style

OUT = C.OUT_DIR / "step3_beijing"


def cv_selected_controls():
    path = C.OUT_DIR / "step2_cv" / "table3_cv_top10.csv"
    if not path.exists():
        import s2_cv_selection
        s2_cv_selection.main()
    return [c.strip() for c in pd.read_csv(path).loc[0, "cities"].split(",")]


def main():
    paper = run_case("beijing", OUT, dict(
        coef="table6_coefficients.csv", effects="table7_effects.csv",
        fig_fit="fig3_pre_fit.png", fig_cf="fig4_counterfactual.png"))

    ours = estimate("beijing", controls=cv_selected_controls())
    rows = []
    for name, r in [("Paper's 7 cities", paper), ("CV(nv)-selected cities (ours)", ours)]:
        m = r["model"]
        rows.append({"control_group": name, "n_controls": len(r["controls"]),
                     "cities": ", ".join(r["controls"]), "pre_R2": m.rsquared,
                     "pre_adj_R2": m.rsquared_adj,
                     "avg_actual": r["post"]["Beijing"].mean(),
                     "avg_counterfactual": r["pred"]["mean"].mean(),
                     "avg_effect": r["avg_effect"], "t_stat": r["t"]})
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(OUT / "table8_control_groups.csv", index=False, float_format="%.4f")

    # Both counterfactuals on one plot
    post = paper["post"]
    x = range(len(post))
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    ax.plot(x, post["Beijing"], color=C.BLUE, lw=2, label="Beijing actual")
    ax.plot(x, paper["pred"]["mean"], color=C.ORANGE, lw=2, ls="--",
            label=f"Paper's 7 cities ({paper['avg_effect']:+.2f} pp)")
    ax.plot(x, ours["pred"]["mean"], color=C.GREY, lw=2, ls=":",
            label=f"CV-selected 5 cities ({ours['avg_effect']:+.2f} pp)")
    ax.set_xticks(list(x)[::2])
    ax.set_xticklabels(post.index[::2], rotation=45, ha="right", fontsize=8)
    ax.set_title("Beijing: counterfactual under two control groups", fontsize=10, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "fig4b_control_groups.png", dpi=250)
    plt.close(fig)

    print("\nControl-group comparison:")
    print(cmp_.drop(columns="cities").to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
