"""Step 5. Chongqing trial property tax (policy start 2011-01-28), paper Section 4.3.

Chongqing has no purchase restriction, so controls are cities without one.

Outputs (outputs/step5_chongqing/):
    table11_coefficients.csv  OLS weights vs paper Table 6
    table12_effects.csv       monthly effects vs paper Table 7
    fig7_pre_fit.png          paper Fig 7
    fig8_counterfactual.png   paper Fig 8
"""
import config as C
from counterfactual import run_case


def main():
    run_case("chongqing", C.OUT_DIR / "step5_chongqing", dict(
        coef="table11_coefficients.csv", effects="table12_effects.csv",
        fig_fit="fig7_pre_fit.png", fig_cf="fig8_counterfactual.png"))


if __name__ == "__main__":
    main()
