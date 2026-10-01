import re
import pandas as pd
import requests
from io import StringIO
import numpy as np



HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}







def get_universe_tickers_simple(market, universe):
    market = str(market).strip().upper()
    universe = str(universe).strip().lower()

    def read_html(url):
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return pd.read_html(StringIO(r.text), header=0, keep_default_na=False)


    def norm_cols(df):
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join(str(x) for x in tup if pd.notna(x)).strip() for tup in df.columns]
        else:
            df.columns = [str(c).strip() for c in df.columns]
        return df

    def first_table_with(tables, cols):
        for df in tables:
            df2 = norm_cols(df)
            if any(c in df2.columns for c in cols):
                return df2
        raise ValueError(f"None of {cols} found in any table.")

    def pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return df[c].astype(str)
        raise ValueError(f"No ticker column found. Columns: {list(df.columns)}")

    def clean_series(s):
        z = s.astype(str).str.strip().str.upper()
        z = z.str.replace(r"\s+", "", regex=True)
        z = z.str.replace("\u200b", "", regex=False)
        z = z.str.replace("–", "-", regex=False)
        z = z.str.replace("·", "-", regex=False)
        return z

    def yahooize_list(lst):
        return sorted(set(x.replace(".", "-") for x in lst if x))

    if market == "US":
        if universe not in {"mega", "sp500", "djia", "ndx"}:
            raise ValueError("US universe must be one of: mega, sp500, djia, ndx")

        out = set()

        if universe in {"mega", "sp500"}:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            df = first_table_with(read_html(url), ["Symbol", "Ticker symbol"])
            out |= set(clean_series(pick_col(df, ["Symbol", "Ticker symbol"])).tolist())

        if universe in {"mega", "djia"}:
            url = "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies"
            df = first_table_with(read_html(url), ["Symbol", "Ticker"])
            out |= set(clean_series(pick_col(df, ["Symbol", "Ticker"])).tolist())

        if universe in {"mega", "ndx"}:
            url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
            df = first_table_with(read_html(url), ["Ticker", "Symbol"])
            out |= set(clean_series(pick_col(df, ["Ticker", "Symbol"])).tolist())

        return yahooize_list(out)

    if market == "CAN":
        if universe not in {"tsx60"}:
            raise ValueError("CAN universe must be: tsx60")
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_60"
        df = first_table_with(read_html(url), ["Symbol", "Ticker"])
        base = clean_series(pick_col(df, ["Symbol", "Ticker"])).tolist()
        base = yahooize_list(base)
        return sorted(set(t + ".TO" for t in base if not t.endswith(".TO")))

    raise ValueError("market must be 'US' or 'CAN'")




def load_tsx_listings_xlsx(file_path, sheet_tsx="TSX Issuers June 2025", sheet_tsxv="TSXV Issuers June 2025", skiprows=9):
    df_tsx  = pd.read_excel(file_path, sheet_name=sheet_tsx,  skiprows=skiprows)
    df_tsxv = pd.read_excel(file_path, sheet_name=sheet_tsxv, skiprows=skiprows)
    df_tsx.columns  = df_tsx.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    df_tsxv.columns = df_tsxv.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    return df_tsx, df_tsxv

def make_universe_tickers_to(df_tsx, df_tsxv=None, universe="tsx60"):
    u = str(universe).lower().strip()
    if u == "tsx60":
        roots = df_tsx[df_tsx["S&P/TSX Index"] == 60]["Root Ticker"].tolist()
    elif u in {"all", "tsx_tsxv"}:
        if df_tsxv is None:
            roots = df_tsx["Root Ticker"].tolist()
        else:
            roots = pd.concat([df_tsx["Root Ticker"], df_tsxv["Root Ticker"]]).dropna().unique().tolist()
    else:
        raise ValueError("universe must be 'tsx60' or 'all'")
    roots = [t.strip().upper() for t in roots if isinstance(t, str) and t.strip()]
    return [f"{t}.TO" for t in roots]



def download_ohlcv_panels(tickers, start="2010-01-01", keep_survivors=True, ffill_prices=True):
    import pandas as pd
    import yfinance as yf

    tickers = [str(t).strip() for t in tickers]
    tickers = [t for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))

    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    buckets = {f: [] for f in fields}
    success = []
    failed = []

    for t in tickers:
        try:
            df = yf.download(t, auto_adjust=True, period="max", progress=False)
            if df is None or df.empty:
                failed.append(t)
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]

            if any(f not in df.columns for f in fields):
                failed.append(t)
                continue

            df = df[fields].copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            for f in fields:
                buckets[f].append(df[[f]].rename(columns={f: t}))

            success.append(t)

        except Exception:
            failed.append(t)

    panels = {}
    for f in fields:
        panels[f] = pd.concat(buckets[f], axis=1) if len(buckets[f]) else pd.DataFrame()

    start_ts = pd.to_datetime(start)
    for f in fields:
        if not panels[f].empty:
            panels[f] = panels[f].loc[panels[f].index >= start_ts]

    if keep_survivors and not panels["Adj Close"].empty:
        adj = panels["Adj Close"]
        keep = adj.columns[~adj.iloc[-1].isna()].tolist()
        removed = [t for t in success if t not in keep]
        failed = list(dict.fromkeys(failed + removed))
        success = keep
        for f in fields:
            if not panels[f].empty:
                panels[f] = panels[f].reindex(columns=keep)

    if ffill_prices:
        for f in ["Open", "High", "Low", "Close", "Adj Close"]:
            if not panels[f].empty:
                panels[f] = panels[f].ffill()

    return panels, success, failed


def load_price_data(indices=None, path="Data Input/prices.parquet"):
    df = pd.read_parquet(path)

    if indices is not None:
        indices = [str(i).strip().lower() for i in indices]
        mask = pd.Series(False, index=df.index)
        for idx in indices:
            col = f"is_{idx}"
            if col not in df.columns:
                available = [c[3:] for c in df.columns if c.startswith("is_")]
                raise ValueError(f"unknown index '{idx}', available: {available}")
            mask = mask | df[col].astype(bool)
        df = df[mask]

    wide = df.pivot(index="date", columns="ticker", values="adj_close")
    wide = wide.sort_index()
    # parquet can store timestamps at ms resolution; pandas 2.x keeps that resolution on
    # read instead of upconverting to [ns], which breaks index equality checks downstream
    # against [ns] indexes built elsewhere (e.g. momentum_backend.add_metadata).
    wide.index = pd.DatetimeIndex(wide.index).as_unit("ns")
    return wide