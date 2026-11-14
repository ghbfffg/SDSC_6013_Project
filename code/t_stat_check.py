"""t-statistic of the average treatment effect, following Li & Bell (2017, J. Econometrics 197),
Theorem 3.2 and eqs. (3.2), (3.3), (3.5):

    T.S. = effect_mean / sqrt( S1 + S2 )
    S1 = xbar' Var(beta_hat) xbar          # = Sigma1_hat / T2, error from the pre-period regression
    S2 = long-run variance of the monthly effects / T2    # = Sigma2_hat / T2

Lag lengths are not given in the paper, so they are set by standard rules:
    regression covariance : Newey-West rule  int(4 * (T1/100)^(2/9))   -> 3 in all three cases
    effect series         : round(T2^(1/4)), the order suggested by Li & Bell -> 2 in all three cases
"""
import numpy as np
import statsmodels.api as sm

import config as C
from s1_build_dataset import load_panel

PAPER_T = {"beijing": -3.8437, "shanghai": 0.0324, "chongqing": -1.5955}


def nw_lag(T):
    return int(4 * (T / 100) ** (2 / 9))


def long_run_var(e, lags, kernel="bartlett"):
    """Eq. (3.3): variance plus autocovariances up to `lags`; Bartlett weights keep it positive."""
    e = np.asarray(e) - np.mean(e)
    n = len(e)
    v = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1) if kernel == "bartlett" else 1.0
        v += 2 * w * (e[k:] @ e[:-k]) / n
    return v


def t_stat(case, xbar_from="post", kernel="bartlett"):
    cfg = C.CASES[case]
    panel = load_panel()
    pre = panel.loc[cfg["pre"][0]:cfg["pre"][1]]
    post = panel.loc[cfg["post"][0]:cfg["post"][1]]
    y, ctrl = cfg["target"], cfg["controls"]

    X_pre = sm.add_constant(pre[ctrl])
    X_post = sm.add_constant(post[ctrl], has_constant="add")
    fit = sm.OLS(pre[y], X_pre).fit()
    effect = (post[y] - fit.predict(X_post)).values

    T1, T2 = len(pre), len(post)
    lag_beta, lag_eff = nw_lag(T1), int(round(T2 ** 0.25))
    V = np.asarray(fit.get_robustcov_results("HAC", maxlags=lag_beta).cov_params())
    xbar = (X_pre if xbar_from == "pre" else X_post).mean().values

    s1 = xbar @ V @ xbar
    s2 = long_run_var(effect, lag_eff, kernel) / T2
    return effect.mean() / np.sqrt(s1 + s2), lag_beta, lag_eff


if __name__ == "__main__":
    print(f"{'case':<10}{'lags':>6}{'paper':>9}{'post-mean, Bartlett':>21}{'pre-mean, rectangular':>23}")
    for case in C.CASES:
        a, lb, le = t_stat(case, "post", "bartlett")
        b, _, _ = t_stat(case, "pre", "rectangular")
        print(f"{case:<10}{f'{lb},{le}':>6}{PAPER_T[case]:>9.4f}{a:>21.4f}{b:>23.4f}")