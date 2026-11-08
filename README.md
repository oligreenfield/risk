# risk

"""
Inflation book direction: 5y-equivalent risk from time-varying, denoised betas,
with a walk-forward backtest and a live (forward) tracker.

Conventions
-----------
* curve   : DataFrame, index = dates, columns = tenors (years, float), values = ZCIS rate in bp
* dv01    : DataFrame, index = dates, columns = tenors, values = P&L per +1bp move in that
            spot tenor's inflation rate (+ = long inflation). Position as at CLOSE of that date.
* 5y-eq   : sum_i dv01_i * beta_i, beta_i = E[dy_i | dy_ref]  (so eq * dy_ref ~ book delta P&L)

Timing (no look-ahead): betas and the ladder known at close t; scored on the move t -> t+1.

Run:  python inflation_direction.py                      (synthetic demo)
      python inflation_direction.py --curve c.csv --dv01 d.csv
"""
from __future__ import annotations

import argparse
import json
import sqlite3

import numpy as np
import pandas as pd

REF = 5.0
TENORS = [1, 2, 3, 5, 7, 10, 15, 20, 30]


# ----------------------------------------------------------------------------
# Covariance / beta estimators.  Each takes (dy_history, ref) -> Series of betas.
# ----------------------------------------------------------------------------
def ewma_cov(dy: pd.DataFrame, halflife: float) -> pd.DataFrame:
    X = dy.values
    w = 0.5 ** (np.arange(len(X))[::-1] / halflife)
    C = (X * w[:, None]).T @ X / w.sum()  # daily changes: zero-mean assumption
    return pd.DataFrame(C, index=dy.columns, columns=dy.columns)


def pca_denoise(cov: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep top-k eigen-factors for the off-diagonals, keep own variances on the diagonal.
    Strips short-end noise (seasonality, illiquidity) out of the cross-tenor covariances."""
    vals, vecs = np.linalg.eigh(cov.values)
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    low = (vecs[:, :k] * vals[:k]) @ vecs[:, :k].T
    low[np.diag_indices_from(low)] = np.diag(cov.values)
    return pd.DataFrame(low, index=cov.index, columns=cov.columns)


def beta_from_cov(cov: pd.DataFrame, ref: float) -> pd.Series:
    return cov[ref] / cov.loc[ref, ref]


def make_estimators(n_static: int = 252, k: int = 3) -> dict:
    def flat(dy, ref):  # naive: 1bp anywhere = 1bp at 5y
        return pd.Series(1.0, index=dy.columns)

    def static(dy, ref):  # frozen historical betas (current practice)
        return beta_from_cov(dy.iloc[:n_static].cov(), ref)

    def rolling(dy, ref):
        return beta_from_cov(dy.iloc[-126:].cov(), ref)

    def ewma(dy, ref):
        return beta_from_cov(ewma_cov(dy, 30), ref)

    def pca(dy, ref):
        return beta_from_cov(pca_denoise(ewma_cov(dy, 30), k), ref)

    def pca_blend(dy, ref):  # fast + slow memory, denoised
        fast = beta_from_cov(pca_denoise(ewma_cov(dy, 20), k), ref)
        slow = beta_from_cov(pca_denoise(ewma_cov(dy, 120), k), ref)
        return 0.5 * fast + 0.5 * slow

    return {"flat_1.0": flat, "static": static, "rolling_126": rolling,
            "ewma_hl30": ewma, "pca_hl30": pca, "pca_blend": pca_blend}


def factor_exposures(dy_hist: pd.DataFrame, dv01: pd.Series, k: int = 3,
                     halflife: float = 60) -> pd.DataFrame:
    """Book exposure to each PCA factor, as 1-sd daily P&L. Shows what the 5y-eq leaves out."""
    cov = ewma_cov(dy_hist, halflife)
    vals, vecs = np.linalg.eigh(cov.values)
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    for j in range(k):  # fix signs: PC1 = parallel up, PC2/3 = long end minus short end
        s = vecs[:, j].sum() if j == 0 else vecs[-1, j] - vecs[0, j]
        vecs[:, j] *= 1 if s >= 0 else -1
    exp = vecs[:, :k].T @ dv01[dy_hist.columns].values
    pnl_sd = exp * np.sqrt(vals[:k])
    return pd.DataFrame({"pnl_1sd": pnl_sd, "var_share": vals[:k] / vals.sum()},
                        index=[f"PC{j + 1}" for j in range(k)])


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def score(df: pd.DataFrame) -> pd.Series:
    """df has columns pred (5y-eq * dy_ref) and real (ladder . dy). Ideal: slope 1, VE high."""
    r, p = df["real"].values, df["pred"].values
    return pd.Series({
        "n": len(df),
        "var_explained": 1 - ((r - p) ** 2).sum() / (r ** 2).sum(),
        "corr": np.corrcoef(r, p)[0, 1] if p.std() > 0 else np.nan,
        "slope": (r @ p) / (p @ p) if (p @ p) > 0 else np.nan,   # 1 = unbiased, <1 = overstated
        "sign_hit": (np.sign(r) == np.sign(p)).mean(),
        "resid_vol": (r - p).std(),
    })


# ----------------------------------------------------------------------------
# Walk-forward backtest
# ----------------------------------------------------------------------------
def backtest(curve: pd.DataFrame, dv01: pd.DataFrame, estimators: dict,
             ref: float = REF, min_hist: int = 252) -> pd.DataFrame:
    dy = curve.diff().dropna()
    dv01 = dv01.reindex(dy.index).ffill()[dy.columns]
    rows = []
    for i in range(min_hist, len(dy) - 1):
        t, nxt = dy.index[i], dy.index[i + 1]
        hist = dy.iloc[: i + 1]                       # information up to close t only
        d, move = dv01.iloc[i], dy.iloc[i + 1]
        real = float((d * move).sum())                # P&L from the full ladder
        for name, est in estimators.items():
            b = est(hist, ref)
            eq = float((d * b).sum())
            rows.append((nxt, name, eq, eq * move[ref], real))
    return pd.DataFrame(rows, columns=["date", "method", "eq_ref", "pred", "real"])


def summarise(bt: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({m: score(g) for m, g in bt.groupby("method")}).T
    return out.sort_values("var_explained", ascending=False)


def plot_rolling(bt: pd.DataFrame, window: int = 120):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    for m, g in bt.groupby("method"):
        g = g.set_index("date")
        num = ((g.real - g.pred) ** 2).rolling(window).sum()
        den = (g.real ** 2).rolling(window).sum()
        (1 - num / den).plot(ax=ax, label=m)
    ax.set_title(f"Rolling {window}d variance explained by 5y-equivalent")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Live (forward) tracker: log forecast at close t, settle on t+1, report rolling stats.
# All methods are logged in parallel (champion / challengers) from the same snapshot.
# ----------------------------------------------------------------------------
class LiveTracker:
    def __init__(self, db_path: str = "live_tracker.db", ref: float = REF, estimators: dict | None = None):
        self.ref = ref
        self.est = estimators or make_estimators()
        self.con = sqlite3.connect(db_path)
        self.con.execute("""CREATE TABLE IF NOT EXISTS forecasts(
            asof TEXT, method TEXT, eq_ref REAL, betas TEXT, dv01 TEXT,
            settle_date TEXT, pred REAL, real REAL, PRIMARY KEY (asof, method))""")

    def forecast(self, asof: pd.Timestamp, dv01: pd.Series, dy_hist: pd.DataFrame) -> pd.DataFrame:
        """Call at close with today's ladder and curve-change history up to and including today."""
        dv01 = dv01[dy_hist.columns]
        out = {}
        for m, est in self.est.items():
            b = est(dy_hist, self.ref)
            eq = float((dv01 * b).sum())
            out[m] = eq
            self.con.execute(
                "INSERT OR REPLACE INTO forecasts(asof, method, eq_ref, betas, dv01) VALUES (?,?,?,?,?)",
                (str(asof.date()), m, eq, json.dumps(b.to_dict()), json.dumps(dv01.to_dict())))
        self.con.commit()
        return pd.Series(out, name="5y_equivalent")

    def settle(self, date: pd.Timestamp, dy_today: pd.Series) -> None:
        """Call when the next close is in: dy_today = move from previous close to this close."""
        asof = self.con.execute(
            "SELECT MAX(asof) FROM forecasts WHERE settle_date IS NULL AND asof < ?",
            (str(date.date()),)).fetchone()[0]
        if asof is None:
            return
        for m, eq, dv in self.con.execute(
                "SELECT method, eq_ref, dv01 FROM forecasts WHERE asof=?", (asof,)).fetchall():
            d = pd.Series(json.loads(dv))
            d.index = d.index.astype(float)
            real = float((d * dy_today[d.index]).sum())
            pred = eq * float(dy_today[self.ref])
            self.con.execute(
                "UPDATE forecasts SET settle_date=?, pred=?, real=? WHERE asof=? AND method=?",
                (str(date.date()), pred, real, asof, m))
        self.con.commit()

    def report(self, window: int = 60) -> pd.DataFrame:
        df = pd.read_sql("SELECT asof, method, pred, real FROM forecasts "
                         "WHERE settle_date IS NOT NULL ORDER BY asof", self.con)
        if df.empty:
            return df
        df = df.groupby("method").tail(window)
        res = pd.DataFrame({m: score(g) for m, g in df.groupby("method")}).T
        base = res.loc["flat_1.0", "var_explained"] if "flat_1.0" in res.index else -np.inf
        res["flag"] = np.where((res.slope < 0.7) | (res.slope > 1.3) | (res.var_explained < base),
                               "CHECK", "ok")
        return res.sort_values("var_explained", ascending=False)


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def load_inputs(curve_csv: str, dv01_csv: str):
    curve = pd.read_csv(curve_csv, index_col=0, parse_dates=True)
    dv01 = pd.read_csv(dv01_csv, index_col=0, parse_dates=True)
    curve.columns = curve.columns.astype(float)
    dv01.columns = dv01.columns.astype(float)
    return curve.sort_index(), dv01.sort_index()


def synthetic(n: int = 1500, seed: int = 0):
    """3-factor curve with vol clustering, a mid-sample slope regime shift, noisy short end,
    and a messy persistent ladder with offsetting positions."""
    rng = np.random.default_rng(seed)
    x = np.array(TENORS, float)
    L = 1.2 - 0.4 * (1 - np.exp(-x / 10))
    S = np.exp(-x / 5) - 0.3
    C = np.exp(-(((x - 7) / 4) ** 2)) - 0.4
    ar = np.zeros((n, 3))
    for t in range(1, n):
        ar[t] = 0.97 * ar[t - 1] + 0.15 * rng.standard_normal(3)
    f = rng.standard_normal((n, 3)) * np.exp(0.5 * ar) * np.array([4.0, 2.0, 1.0])
    S_t = np.where(np.arange(n)[:, None] < n // 2, S, -1.5 * S)  # regime shift
    dy = f[:, [0]] * L + f[:, [1]] * S_t + f[:, [2]] * C
    dy += rng.standard_normal((n, len(x))) * (0.3 + 1.5 * np.exp(-x / 1.5))
    dates = pd.bdate_range("2020-01-01", periods=n + 1)
    curve = pd.DataFrame(np.vstack([np.full(len(x), 350.0), 350 + dy.cumsum(0)]), index=dates, columns=x)
    lad = np.zeros((n + 1, len(x)))
    for t in range(1, n + 1):
        lad[t] = 0.92 * lad[t - 1] + rng.standard_normal(len(x)) * 8
    return curve, pd.DataFrame(lad, index=dates, columns=x)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve")
    ap.add_argument("--dv01")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    curve, dv01 = load_inputs(a.curve, a.dv01) if a.curve and a.dv01 else synthetic()
    ests = make_estimators()
    dy = curve.diff().dropna()

    print("\n=== Backtest (walk-forward, out-of-sample) ===")
    bt = backtest(curve, dv01, ests)
    print(summarise(bt).round(3).to_string())
    if a.plot:
        plot_rolling(bt)

    print("\n=== Live-tracker simulation over last 40 days ===")
    tracker = LiveTracker(":memory:", estimators=ests)
    for date in dy.index[-40:]:
        tracker.settle(date, dy.loc[date])                       # score yesterday's forecast
        tracker.forecast(date, dv01.loc[date], dy.loc[:date])    # log today's forecast
    print(tracker.report(window=40).round(3).to_string())

    print("\n=== Today's book ===")
    last = dy.index[-1]
    print(tracker.forecast(last, dv01.loc[last], dy).round(1).to_string())
    print("\nFactor exposures (1-sd daily P&L):")
    print(factor_exposures(dy, dv01.loc[last]).round(2).to_string())
