"""Step 1. Build the monthly panel of year-on-year new-house price growth.

The NBS price index is not available as one continuous file, so three sources are spliced:

    2008-01 .. 2008-12  EPS export (the only source covering early 2008)
    2009-01 .. 2009-02  CME quarterly file, monthly rows (Fresgn == "M")
    2009-03 onwards     RERR 70-city file

Each month is taken from exactly one source. Index 104.2 means +4.2% vs the same
month a year ago, so growth = index - 100.

Only the "ResidentIDX" column of RERR is used. From 2011 NBS also publishes a
separate commodity-housing index whose values differ (Beijing 2011-02: 106.8 vs
108.4); the paper's Beijing value of 6.8% matches ResidentIDX.

Outputs (outputs/step1_data/):
    table1_data_sources.csv    what each source contributes
    table2_overlap_check.csv   agreement between sources where they overlap
    fig1_data_coverage.png     source used for each month
    panel_yoy_growth.csv       final panel (month x city), growth in %
"""
import re
import warnings

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

warnings.filterwarnings("ignore", category=UserWarning)
OUT = C.OUT_DIR / "step1_data"

EPS_FILES = ["eps_2008_monthly_1.xlsx", "eps_2008_monthly_2.xlsx"]
EPS_INDICATOR = "新建商品住宅销售价格指数（上年同月=100）"
GAP_MONTHS = ["2009-01", "2009-02"]          # missing in both EPS and RERR


def english(name):
    """'北京市' -> 'Beijing'; None if the city is not one we use."""
    return C.CITY_NAMES.get(re.sub(r"(市|地区)$", "", str(name).strip()))


def load_eps():
    frames = []
    for f in EPS_FILES:
        raw = pd.read_excel(C.RAW_DIR / f)
        raw = raw[(raw.iloc[:, 0] == EPS_INDICATOR) & (raw.iloc[:, 2] == "本期数")]
        month_cols = [c for c in raw.columns[3:] if re.match(r"^\d{2}-\d{4}$", str(c))]
        long = raw.melt(id_vars=[raw.columns[1]], value_vars=month_cols,
                        var_name="m", value_name="index")
        long["city"] = long[raw.columns[1]].map(english)
        long["month"] = long["m"].map(lambda s: f"{s[3:]}-{s[:2]}")   # '01-2008' -> '2008-01'
        frames.append(long[["city", "month", "index"]])
    # The two exports list the same cities with identical values (checked), so drop repeats.
    return pd.concat(frames).dropna().drop_duplicates()


def load_rerr(last_month="2013-06"):
    raw = pd.read_excel(C.RAW_DIR / "rerr_70city_2008_2024.xlsx",
                        sheet_name="sheet1", skiprows=[1]).iloc[1:]
    d = raw[(raw["FixedBase"] == "同比") & (raw["AreaName"] != "中国")]
    d = d[(d["SgnMonth"] >= "2008-11") & (d["SgnMonth"] <= last_month)]
    return pd.DataFrame({"city": d["AreaName"].map(english),
                         "month": d["SgnMonth"], "index": d["ResidentIDX"]})


def load_cme():
    """Monthly rows (Fresgn == 'M') of the quarterly file; Epiq0502 = new-house index."""
    raw = pd.read_excel(C.RAW_DIR / "cme_quarterly_2004_2011.xlsx",
                        sheet_name="sheet1", skiprows=[1]).iloc[1:]
    d = raw[raw["Fresgn"] == "M"]
    return pd.DataFrame({"city": d["Citynm"].map(english),
                         "month": d["Staper"], "index": d["Epiq0502"]})


def tidy(df, source):
    df = df.dropna(subset=["city", "index"]).copy()
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    df["source"] = source
    return df.dropna(subset=["index"])


def overlap_check(a, b, name_a, name_b):
    """Compare two sources on the city-months they share."""
    m = a.merge(b, on=["city", "month"], suffixes=("_a", "_b"))
    if m.empty:
        return None
    diff = (m["index_a"] - m["index_b"]).abs()
    return {"source_a": name_a, "source_b": name_b, "n_overlap": len(m),
            "months": f"{m['month'].min()} ~ {m['month'].max()}",
            "share_identical": round((diff < 1e-9).mean(), 3),
            "max_abs_diff": round(diff.max(), 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eps, rerr, cme = tidy(load_eps(), "EPS"), tidy(load_rerr(), "RERR"), tidy(load_cme(), "CME")

    # --- splice ---------------------------------------------------------
    long = pd.concat([eps[eps["month"] <= "2008-12"],
                      cme[cme["month"].isin(GAP_MONTHS)],
                      rerr[rerr["month"] >= "2009-03"]])
    panel = (long.pivot(index="month", columns="city", values="index") - 100).sort_index()
    panel.to_csv(OUT / "panel_yoy_growth.csv", float_format="%.1f")

    # --- Table 1: sources -----------------------------------------------
    used = long.groupby("source")["month"].agg(["min", "max", "count"])
    t1 = pd.DataFrame([
        ["EPS (2 exports)", "2008-01 ~ 2008-12", "Early 2008; only source for these months"],
        ["CME quarterly file (monthly rows)", "2009-01 ~ 2009-02", "Fills the gap between EPS and RERR"],
        ["RERR 70-city file", "2009-03 ~ 2013-06", "Main source; column ResidentIDX"],
    ], columns=["source", "months_used", "role"])
    t1["city_months_used"] = [int(used.loc[s, "count"]) for s in ["EPS", "CME", "RERR"]]
    t1.to_csv(OUT / "table1_data_sources.csv", index=False)

    # --- Table 2: overlap consistency ----------------------------------
    checks = [overlap_check(eps, rerr, "EPS", "RERR"), overlap_check(cme, rerr, "CME", "RERR")]
    t2 = pd.DataFrame([c for c in checks if c])
    t2.to_csv(OUT / "table2_overlap_check.csv", index=False)

    # --- Figure 1: which source supplies each month ----------------------
    months = list(panel.index)
    src_by_month = long.groupby("month")["source"].agg(lambda s: s.mode()[0]).reindex(months)
    colors = {"EPS": C.BLUE, "CME": C.ORANGE, "RERR": C.GREY}
    fig, ax = plt.subplots(figsize=(9, 2.2), layout="constrained")
    for i, m in enumerate(months):
        ax.bar(i, 1, width=0.9, color=colors[src_by_month[m]])
    for s, c in colors.items():
        ax.bar(0, 0, color=c, label=s)
    ax.set_xticks(range(0, len(months), 6))
    ax.set_xticklabels(months[::6], rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_title("Data source used for each month (2008-01 to 2013-06)", fontsize=10, loc="left", pad=22)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.savefig(OUT / "fig1_data_coverage.png", dpi=250)
    plt.close(fig)

    print(t1.to_string(index=False))
    print("\nOverlap check:\n", t2.to_string(index=False))
    print(f"\nPanel: {panel.shape[0]} months x {panel.shape[1]} cities, "
          f"missing values: {int(panel.isna().sum().sum())}")


def load_panel():
    """Used by later steps: month x city table of growth rates (%)."""
    path = OUT / "panel_yoy_growth.csv"
    if not path.exists():
        main()
    return pd.read_csv(path, index_col="month")


if __name__ == "__main__":
    main()
