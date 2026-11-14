# Replication of Du & Zhang (2015)

"Home-purchase restriction, property tax and housing price in China: A counterfactual analysis", Journal of Econometrics 188, 558-568.

## Run

```bash
cd code
python run_all.py          # steps 1-6, a few minutes
python s7_monte_carlo.py --reps 20   # step 7, several hours; resumable, run separately
python s3_beijing.py       # or any single step
```

Requires pandas, numpy, statsmodels, scipy, matplotlib, openpyxl. Raw data are in `data/raw/`. Each step writes its tables and figures to `outputs/stepN_*/`.

## Files

| File | Content | Outputs |
|---|---|---|
| `config.py` | paths, city names, control groups and sample windows | |
| `paper_values.py` | numbers printed in the paper, used only for checking | |
| `s1_build_dataset.py` | splice EPS, CME and RERR into one monthly panel | Tables 1-2, Fig 1 |
| `s2_cv_selection.py` | CV(nv) control-city search (paper Section 3.1) | Tables 3-5, Fig 2 |
| `s3_beijing.py` | Beijing purchase restriction (paper Section 4.1) | Tables 6-8, Figs 3-4 |
| `s4_shanghai.py` | Shanghai property tax (Section 4.2) | Tables 9-10, Figs 5-6 |
| `s5_chongqing.py` | Chongqing property tax (Section 4.3) | Tables 11-12, Figs 7-8 |
| `s6_summary.py` | three cases against the paper | Table 13, Fig 9 |
| `s7_monte_carlo.py` | Monte Carlo: CV(nv) vs AICC vs AIC (paper Section 3.2, Table 1) | Table 14, Fig 10 |
| `counterfactual.py` | shared OLS, counterfactual, t-test and plotting code | |

## Notes

- Growth rate = year-on-year price index - 100.
- The t-statistic follows Li & Bell (2017, Theorem 3.2): a Newey-West covariance of the regression coefficients plus a Bartlett long-run variance of the monthly effects. Lag lengths come from standard rules (3 and 2 in all three cases); the regressor mean is taken over the post-policy months, which is what reproduces the paper's values.
- The monthly 95% band is the OLS confidence interval of the counterfactual mean. The paper does not describe how its band is built.
- The main results use the control groups reported in the paper. Step 2 searches the 21 Beijing candidates again, and step 3 compares the result with the paper's group.
- Step 7 is not part of `run_all.py` because it takes hours. Finished replications are saved in `outputs/step7_monte_carlo/raw_results.csv`; rerun the command to continue an interrupted run, or `python s7_monte_carlo.py --summary-only` to rebuild the tables.
- The paper uses 100 replications per setting; we use 20 for time reasons, so differences between methods in the large-sample settings are small relative to the noise.
