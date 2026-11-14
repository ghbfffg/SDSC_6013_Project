"""Step 4. Shanghai trial property tax (policy start 2011-01-28), paper Section 4.2.

Controls are cities that adopted purchase restrictions around the same time as Shanghai,
so the comparison isolates the property tax from the purchase restriction.

Outputs (outputs/step4_shanghai/):
    table9_coefficients.csv   OLS weights vs paper Table 4
    table10_effects.csv       monthly effects vs paper Table 5
    fig5_pre_fit.png          paper Fig 5
    fig6_counterfactual.png   paper Fig 6
"""
import config as C
from counterfactual import run_case


def main():
    run_case("shanghai", C.OUT_DIR / "step4_shanghai", dict(
        coef="table9_coefficients.csv", effects="table10_effects.csv",
        fig_fit="fig5_pre_fit.png", fig_cf="fig6_counterfactual.png"))


if __name__ == "__main__":
    main()
