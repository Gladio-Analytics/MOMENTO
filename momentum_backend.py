import time
import numpy as np
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from tqdm.auto import tqdm


def fetch_prices(tickers, keep_survivors=True, ffill_prices=True, max_retries=3, retry_backoff=5.0):
    tickers = list(dict.fromkeys([str(t).strip().upper() for t in tickers if t]))
    #start_ts = pd.to_datetime(start)

    frames = []
    success = []
    failed = []
    fail_info = {}

    for t in tqdm(tickers, total=len(tickers)):
        try:
            # Shared cloud IPs (e.g. Streamlit Community Cloud) get Yahoo-rate-limited far more
            # easily than a home connection, so a transient 429 shouldn't sink the whole fetch.
            df = None
            for attempt in range(max_retries):
                try:
                    df = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
                    break
                except YFRateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(retry_backoff * (attempt + 1))
            if df is None or df.empty:
                failed.append(t)
                fail_info[t] = "empty"
                continue

            if isinstance(df.columns, pd.MultiIndex):
                lv0 = df.columns.get_level_values(0)
                lv1 = df.columns.get_level_values(1)

                if ("Adj Close" in lv0) or ("Close" in lv0):
                    field = "Adj Close" if ("Adj Close" in lv0) else "Close"
                    block = df[field]
                    if isinstance(block, pd.DataFrame):
                        s = block[t] if (t in block.columns) else block.iloc[:, 0]
                    else:
                        s = block
                elif ("Adj Close" in lv1) or ("Close" in lv1):
                    field = "Adj Close" if ("Adj Close" in lv1) else "Close"
                    block = df.xs(field, level=1, axis=1)
                    s = block[t] if (t in block.columns) else block.iloc[:, 0]
                else:
                    failed.append(t)
                    fail_info[t] = "no Adj Close/Close"
                    continue
            else:
                if "Adj Close" in df.columns:
                    s = df["Adj Close"]
                elif "Close" in df.columns:
                    s = df["Close"]
                else:
                    failed.append(t)
                    fail_info[t] = "no Adj Close/Close"
                    continue

            s = s.copy()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            #s = s.loc[s.index >= start_ts]
            s = s.dropna()

            if s.empty:
                failed.append(t)
                #fail_info[t] = "no data after start"
                continue

            frames.append(s.rename(t))
            success.append(t)

        except Exception as e:
            failed.append(t)
            fail_info[t] = str(e)

    prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()

    if prices.empty:
        prices = pd.DataFrame()
        prices.attrs["fail_info"] = fail_info
        return prices, [], failed

    prices = prices.sort_index()

    if keep_survivors:
        last_dt = prices.index.max()
        keep = prices.columns[prices.loc[last_dt].notna()].tolist()
        dropped = [c for c in prices.columns if c not in keep]
        for d in dropped:
            fail_info[d] = "not survivor"
        prices = prices[keep]
        success = keep
        failed = list(dict.fromkeys(failed + dropped))

    if ffill_prices:
        prices = prices.ffill()

    prices.attrs["fail_info"] = fail_info
    return prices, success, failed


def prep_reference(reference_close, reference_tickers):
    ref = reference_close.copy()
    ref.columns = [str(c).strip().upper() for c in ref.columns]

    rt = [str(t).strip().upper() for t in reference_tickers]
    if len(rt) != 3:
        raise ValueError("reference_tickers must be length 3: [passive_bench, active_bench, riskfree]")

    passive, active, rf = rt[0], rt[1], rt[2]

    if passive not in ref.columns:
        raise KeyError(f"Missing passive benchmark ticker in reference_close: {passive}")
    if active not in ref.columns:
        raise KeyError(f"Missing active benchmark ticker in reference_close: {active}")
    if rf not in ref.columns:
        raise KeyError(f"Missing riskfree ticker in reference_close: {rf}")

    s_passive = ref[passive].copy()
    s_active  = ref[active].copy()
    s_rf      = ref[rf].copy()

    out = pd.concat(
        [
            s_active.rename(f"Active Benchmark ({active})"),
            s_passive.rename(f"Passive Benchmark ({passive})"),
            s_rf.rename(f"Risk-Free Rate ({rf})"),
        ],
        axis=1,
    )

    return out


def check_gaps(df, fill_na_gaps=False):
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        try:
            d.index = pd.to_datetime(d.index)
        except Exception:
            pass
    if not d.index.is_monotonic_increasing:
        d = d.sort_index()

    per_col = {}
    holes_total = 0
    leading_total = 0
    trailing_total = 0
    total_na = int(d.isna().sum().sum())

    for c in d.columns:
        s = d[c]
        fv = s.first_valid_index()
        lv = s.last_valid_index()

        if fv is None or lv is None:
            leading = int(s.isna().sum())
            trailing = 0
            holes = 0
            span_len = 0
        else:
            between = s.loc[fv:lv]
            holes = int(between.isna().sum())
            leading = int(s.loc[:fv].isna().sum())
            trailing = int(s.loc[lv:].isna().sum())
            span_len = int(len(between))

        per_col[c] = dict(
            total_na=int(s.isna().sum()),
            leading_na=leading,
            trailing_na=trailing,
            holes_na=holes,
            first_valid=fv,
            last_valid=lv,
            span_len=span_len,
        )

        holes_total += holes
        leading_total += leading
        trailing_total += trailing

    gap_report = dict(
        per_column=per_col,
        df=dict(
            total_na=total_na,
            leading_na=leading_total,
            trailing_na=trailing_total,
            holes_na=holes_total,
        ),
    )

    filled_per_col = {c: 0 for c in d.columns}
    filled_total = 0

    if fill_na_gaps:
        out = d.copy()
        idx = out.index

        for c in out.columns:
            s = out[c]
            fv = s.first_valid_index()
            lv = s.last_valid_index()
            if fv is None or lv is None:
                continue

            mask = (idx >= fv) & (idx <= lv)
            before = int(s.loc[mask].isna().sum())

            tmp = s.copy()
            tmp.loc[~mask] = np.nan
            tmp = tmp.ffill()

            out.loc[mask, c] = tmp.loc[mask]
            after = int(out.loc[mask, c].isna().sum())

            filled = max(0, before - after)
            filled_per_col[c] = int(filled)
            filled_total += int(filled)
    else:
        out = d

    return dict(
        gap_report=gap_report,
        filled=bool(fill_na_gaps),
        df=out,
        filled_per_column=filled_per_col,
        filled_total=int(filled_total),
    )


def get_available_sample(adj_close, reference_close):
    a = adj_close.copy()
    r = reference_close.copy()
    a.index = pd.to_datetime(a.index)
    r.index = pd.to_datetime(r.index)
    return a.index.intersection(r.index).sort_values()


def get_usable_sample(available_sample, start):
    idx = pd.DatetimeIndex(pd.to_datetime(available_sample))
    start_dt = pd.to_datetime(start)
    return idx[idx >= start_dt].sort_values()


def trim_adj_close_prices(df, usable_sample):

    d = df.copy()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    us = pd.DatetimeIndex(pd.to_datetime(usable_sample)).sort_values().unique()
    if len(us) == 0:
        return d.iloc[0:0].copy()
    first = us[0]
    pos = d.index.searchsorted(first, side="left")
    prev_idx = d.index[pos - 1] if pos > 0 else None
    keep = us.intersection(d.index)
    if prev_idx is not None:
        keep = keep.union(pd.DatetimeIndex([prev_idx]))

    keep = keep.sort_values()
    return d.loc[keep]


def to_returns(px):
    out = px.pct_change()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.iloc[1:].copy()
    return out


def add_metadata(df, metadata, how="left", require_same_index=True):

    if df is None or metadata is None:
        raise ValueError("df and metadata must be provided")

    d = df.copy()
    m = metadata.copy()

    d.index = pd.to_datetime(d.index)
    m.index = pd.to_datetime(m.index)

    if require_same_index:
        if not d.index.equals(m.index):
            # helpful diagnostics
            only_in_meta = m.index.difference(d.index)
            only_in_df = d.index.difference(m.index)
            raise ValueError(
                "Index mismatch between metadata and df.\n"
                f"metadata rows={len(m.index)} df rows={len(d.index)}\n"
                f"only_in_metadata (first 5)={list(only_in_meta[:5])}\n"
                f"only_in_df (first 5)={list(only_in_df[:5])}"
            )

    # metadata is left
    out = m.join(d, how=how)

    # if you forced same-index, this should never happen
    if require_same_index and len(out) != len(m):
        raise ValueError("Join changed row count unexpectedly (check index uniqueness).")

    return out


def get_t0(metadata, holdings_by_slot_daily, exec_col=None, slot_col="Top_1"):
    md = metadata.copy()
    md.index = pd.to_datetime(md.index)
    md = md.sort_index()

    hs = holdings_by_slot_daily.copy()
    hs.index = pd.to_datetime(hs.index)
    hs = hs.sort_index()

    if exec_col is None:
        if "is_actual_execution_day" in md.columns:
            exec_col = "is_actual_execution_day"
        elif "is_execution_day" in md.columns:
            exec_col = "is_execution_day"
        else:
            raise KeyError("metadata must contain 'is_actual_execution_day' or 'is_execution_day'")

    if slot_col not in hs.columns:
        top_cols = [c for c in hs.columns if str(c).startswith("Top_")]
        if len(top_cols) == 0:
            raise KeyError("holdings_by_slot_daily must contain Top_* columns")
        top_cols = sorted(top_cols, key=lambda c: int(str(c).split("_", 1)[1]) if str(c).split("_", 1)[1].isdigit() else 10**9)
        slot_col = top_cols[0]

    idx = md.index.intersection(hs.index).sort_values()
    if len(idx) == 0:
        return pd.NaT

    e = md.loc[idx, exec_col].astype(bool)
    s = hs.loc[idx, slot_col].map(lambda x: isinstance(x, str) and len(x) > 0)
    m = e & s

    return idx[m.argmax()] if m.any() else pd.NaT


def populate_portfolio(metadata, picks_df, universe=None):
    md = metadata.copy()
    md.index = pd.to_datetime(md.index)
    md = md.sort_index()

    pk = picks_df.copy()
    if "execution_date" not in pk.columns:
        raise KeyError("picks_df must contain column 'execution_date'")

    slot_cols = [c for c in pk.columns if str(c).startswith("Top_")]
    if len(slot_cols) == 0:
        raise ValueError("picks_df has no Top_* columns")

    def _slot_key(c):
        s = str(c)
        n = s.split("_", 1)[1] if "_" in s else ""
        try:
            return int(n)
        except Exception:
            return 10**9

    slot_cols = sorted(slot_cols, key=_slot_key)

    pk["execution_date"] = pd.to_datetime(pk["execution_date"])
    pk = pk.loc[pk["execution_date"].notna()].copy()

    if "is_effective_signal" in pk.columns:
        pk = pk.loc[pd.Series(pk["is_effective_signal"]).astype(bool).values].copy()

    pk = pk.loc[pd.Series(pk[slot_cols[0]]).notna().values].copy()
    if len(pk) == 0:
        raise ValueError("No usable rows in picks_df after filtering (execution_date/effective/top1)")

    pk = pk.sort_values("execution_date").copy()
    pk = pk.set_index("execution_date", drop=False)

    if pk.index.has_duplicates:
        pk = pk[~pk.index.duplicated(keep="last")]

    if universe is None:
        vals = []
        for c in slot_cols:
            vals.extend([x for x in pk[c].tolist() if isinstance(x, str) and x])
        universe = sorted(set(vals))

    universe = [str(x) for x in list(universe)]
    if len(universe) == 0:
        raise ValueError("universe is empty")

    daily_idx = md.index
    ex_dates = pk.index.intersection(daily_idx).sort_values()
    if len(ex_dates) == 0:
        raise ValueError("No execution dates from picks_df fall on metadata daily index")

    slot_daily = pd.DataFrame(index=daily_idx, columns=slot_cols, dtype=object)
    slot_daily[:] = np.nan

    uni = pd.DataFrame(index=daily_idx, columns=universe, dtype=object)
    uni[:] = np.nan
    colpos = {t: j for j, t in enumerate(universe)}

    for i, exd in enumerate(ex_dates):
        i0 = daily_idx.searchsorted(exd)
        if i + 1 < len(ex_dates):
            i1 = daily_idx.searchsorted(ex_dates[i + 1])
        else:
            i1 = len(daily_idx)
        if i1 <= i0:
            continue

        row = pk.loc[exd, slot_cols]

        slot_daily.iloc[i0:i1, :] = row.values

        held = []
        for c in slot_cols:
            x = row[c]
            if isinstance(x, str) and x:
                held.append(x)

        seen = set()
        held_u = []
        for t in held:
            if t in seen:
                continue
            seen.add(t)
            held_u.append(t)

        missing = [t for t in held_u if t not in colpos]
        if len(missing) > 0:
            raise ValueError(f"Found tickers in picks_df not in universe: {missing[:10]}")

        if len(held_u) > 0:
            cols_j = [colpos[t] for t in held_u]
            uni.iloc[i0:i1, cols_j] = True

    holdings_universe_daily = pd.concat([md, uni], axis=1)
    holdings_by_slot_daily = pd.concat([md, slot_daily], axis=1)

    return holdings_universe_daily, holdings_by_slot_daily


def populate_portfolio_returns(ret_ac, metadata, holdings_universe_daily, holdings_by_slot_daily):
    md = metadata.copy()
    md.index = pd.to_datetime(md.index)
    md = md.sort_index()
    idx = md.index

    r = ret_ac.copy()
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()
    r = r.reindex(idx)

    meta_cols = list(md.columns)

    hu = holdings_universe_daily.copy()
    hu.index = pd.to_datetime(hu.index)
    hu = hu.reindex(idx)

    hs = holdings_by_slot_daily.copy()
    hs.index = pd.to_datetime(hs.index)
    hs = hs.reindex(idx)

    hold_cols = [c for c in hu.columns if c not in meta_cols]
    if len(hold_cols) == 0:
        raise ValueError("holdings_universe_daily has no non-metadata ticker columns")

    missing_ret = [c for c in hold_cols if c not in r.columns]
    if len(missing_ret) > 0:
        raise ValueError(f"ret_ac is missing tickers present in holdings_universe_daily: {missing_ret[:10]}")

    uni_mask = hu[hold_cols].notna()
    uni_ret = r[hold_cols].where(uni_mask)

    slot_cols = [c for c in hs.columns if str(c).startswith("Top_")]
    if len(slot_cols) == 0:
        raise ValueError("holdings_by_slot_daily has no Top_* slot columns")

    out_slot = pd.DataFrame(index=idx, columns=slot_cols, dtype=float)

    r_cols = pd.Index(r.columns)
    r_vals = r.values
    n = len(idx)
    rows = np.arange(n)

    for c in slot_cols:
        t = hs[c].astype(object).values
        col_idx = np.full(n, -1, dtype=int)

        m = pd.notna(t)
        if m.any():
            t_str = np.array([str(x) for x in t[m]], dtype=object)
            got = r_cols.get_indexer(t_str)
            if (got < 0).any():
                bad = sorted(set(t_str[got < 0].tolist()))
                raise ValueError(f"Slot {c} contains tickers not in ret_ac columns: {bad[:10]}")
            col_idx[m] = got

        vals = np.full(n, np.nan, dtype=float)
        ok = col_idx >= 0
        if ok.any():
            vals[ok] = r_vals[rows[ok], col_idx[ok]]

        out_slot[c] = vals

    returns_universe_daily = pd.concat([md, uni_ret], axis=1)
    returns_by_slot_daily = pd.concat([md, out_slot], axis=1)

    return returns_universe_daily, returns_by_slot_daily


def get_buy_and_sell_df(metadata, holdings_by_slot_daily, top_k, slippage_bps=0.0):
    md = metadata.copy()
    md.index = pd.to_datetime(md.index)
    md = md.sort_index()

    hs = holdings_by_slot_daily.copy()
    hs.index = pd.to_datetime(hs.index)
    hs = hs.sort_index()

    K = int(top_k)
    if K <= 0:
        raise ValueError("top_k must be >= 1")

    slip = float(slippage_bps)

    slot_cols = [c for c in hs.columns if str(c).startswith("Top_")]
    if len(slot_cols) == 0:
        raise ValueError("holdings_by_slot_daily must contain Top_* columns")

    def _slot_key(c):
        s = str(c)
        n = s.split("_", 1)[1] if "_" in s else ""
        try:
            return int(n)
        except Exception:
            return 10**9

    slot_cols = sorted(slot_cols, key=_slot_key)
    slot_cols = slot_cols[:K]

    exec_flag = "is_actual_execution_day" if "is_actual_execution_day" in md.columns else "is_execution_day"
    if exec_flag not in md.columns:
        raise KeyError("metadata must contain is_actual_execution_day or is_execution_day")

    sig_col = "actual_signal_date" if "actual_signal_date" in md.columns else "signal_date"
    if sig_col not in md.columns:
        raise KeyError("metadata must contain actual_signal_date or signal_date")

    idx = md.index.intersection(hs.index)
    if len(idx) == 0:
        raise ValueError("No overlapping dates between metadata and holdings_by_slot_daily")

    md = md.loc[idx]
    hs = hs.loc[idx]

    exec_dates = pd.DatetimeIndex(md.index[md[exec_flag].astype(bool).values]).sort_values()
    if len(exec_dates) == 0:
        raise ValueError(f"No execution days found using {exec_flag}")

    buy_cols = [f"buy_{i}" for i in range(1, K + 1)]
    sell_cols = [f"sell_{i}" for i in range(1, K + 1)]

    buys_rows = []
    sells_rows = []
    out_index = []

    def _row_to_list(row):
        out = []
        for x in row:
            if isinstance(x, str) and x:
                out.append(x)
        return out

    for exd in exec_dates:
        pos = idx.searchsorted(exd)
        if pos < 0 or pos >= len(idx) or idx[pos] != exd:
            continue

        new_row = hs.loc[exd, slot_cols]
        new_list = _row_to_list(new_row.values)

        if pos == 0:
            old_list = []
        else:
            prev_dt = idx[pos - 1]
            old_row = hs.loc[prev_dt, slot_cols]
            old_list = _row_to_list(old_row.values)

        old_set = set(old_list)
        new_set = set(new_list)

        buys = []
        for t in new_list:
            if t not in old_set and t not in buys:
                buys.append(t)

        sells = []
        for t in old_list:
            if t not in new_set and t not in sells:
                sells.append(t)

        n_buy = int(len(buys))
        n_sell = int(len(sells))

        buy_pad = buys[:K] + [np.nan] * max(0, K - len(buys))
        sell_pad = sells[:K] + [np.nan] * max(0, K - len(sells))

        sig_dt = pd.to_datetime(md.loc[exd, sig_col])

        buys_rows.append(
            [exd, sig_dt, n_buy, (slip * n_buy)] + buy_pad
        )
        sells_rows.append(
            [exd, sig_dt, n_sell, (slip * n_sell)] + sell_pad
        )
        out_index.append(exd)

    if len(out_index) == 0:
        raise ValueError("No buy/sell rows produced")

    buys_df = pd.DataFrame(
        buys_rows,
        columns=["execution_date", "signal_date_associated", "n_buys", "cost_bps"] + buy_cols,
    )
    sells_df = pd.DataFrame(
        sells_rows,
        columns=["execution_date", "signal_date_associated", "n_sells", "cost_bps"] + sell_cols,
    )

    buys_df["execution_date"] = pd.to_datetime(buys_df["execution_date"])
    buys_df["signal_date_associated"] = pd.to_datetime(buys_df["signal_date_associated"])
    sells_df["execution_date"] = pd.to_datetime(sells_df["execution_date"])
    sells_df["signal_date_associated"] = pd.to_datetime(sells_df["signal_date_associated"])

    buys_df = buys_df.sort_values("execution_date").set_index("execution_date", drop=False)
    sells_df = sells_df.sort_values("execution_date").set_index("execution_date", drop=False)

    return buys_df, sells_df


def extract_core_portfolio_series(daily):
    if not isinstance(daily, dict) or "core" not in daily:
        raise KeyError("daily must be a dict containing key 'core'")

    core = daily["core"].copy()
    core.index = pd.to_datetime(core.index)
    core = core.sort_index()

    if "locked_cash" not in core.columns:
        raise KeyError("daily['core'] must contain 'locked_cash'")

    # locked_cash is a circulating reinvest-cash balance (replaced each execution day based on
    # that day's full cash flow, not accumulated), so "injection" is simply its day-over-day
    # change: positive when the balance grew, negative when a prior balance got deployed.
    core["locked_cash_injection"] = core["locked_cash"].astype(float).diff().fillna(core["locked_cash"].astype(float))

    need = ["locked_cash", "locked_cash_injection", "equity_value_end", "stocks_plus_cash", "equity_return"]
    miss = [c for c in need if c not in core.columns]
    if len(miss) > 0:
        raise KeyError(f"daily['core'] is missing columns: {miss}")

    return core


def extract_notional_series(daily):
    """Pull the ratcheting current_notional watermark and related sleeve values out of a
    simulate_portfolio_ledger run, for tracking/plotting how the reinvestment floor has grown
    over time relative to actual portfolio value and how much has been permanently skimmed off.
    """
    if not isinstance(daily, dict) or "core" not in daily:
        raise KeyError("daily must be a dict containing key 'core'")

    core = daily["core"].copy()
    core.index = pd.to_datetime(core.index)
    core = core.sort_index()

    need = ["current_notional", "stocks_plus_cash", "bench_bucket", "rf_bucket"]
    miss = [c for c in need if c not in core.columns]
    if len(miss) > 0:
        raise KeyError(f"daily['core'] is missing columns: {miss}")

    out = pd.DataFrame(index=core.index)
    out["current_notional"] = core["current_notional"].astype(float)
    out["stocks_plus_cash"] = core["stocks_plus_cash"].astype(float)
    out["bench_bucket"] = core["bench_bucket"].astype(float)
    out["rf_bucket"] = core["rf_bucket"].astype(float)
    out["total_skimmed"] = out["bench_bucket"] + out["rf_bucket"]

    # unrealized profit sitting above the watermark - not yet "locked in" by a ratchet step
    out["gap_above_notional"] = (out["stocks_plus_cash"] - out["current_notional"]).clip(lower=0.0)

    # how much the watermark itself grew each day (0 on most days, positive on a ratchet step)
    out["notional_ratchet_step"] = out["current_notional"].diff().fillna(0.0).clip(lower=0.0)

    return out


def create_strategy_value(core_series, ret_ref_with_meta):
    """Attach benchmark/risk-free return columns and expose the ledger's own cash-sleeve
    tracking under the column names the reporting/viz layer expects.

    simulate_portfolio_ledger already compounds the bench/rf cash sleeves itself (via
    cash_alloc_bench/cash_alloc_rf) and returns them as core_series["bench_bucket"]/
    ["rf_bucket"] - this function does not re-simulate that split, it only surfaces it.
    """
    cs = core_series.copy()
    cs.index = pd.to_datetime(cs.index)
    cs = cs.sort_index()

    rr = ret_ref_with_meta.copy()
    rr.index = pd.to_datetime(rr.index)
    rr = rr.sort_index()

    need_cs = ["locked_cash", "bench_bucket", "rf_bucket", "cash_added_today", "equity_value_end", "stocks_plus_cash"]
    miss = [c for c in need_cs if c not in cs.columns]
    if len(miss) > 0:
        raise KeyError(f"core_series is missing columns: {miss}")

    cols = list(rr.columns)
    active_cols = [c for c in cols if str(c).startswith("Active Benchmark (")]
    passive_cols = [c for c in cols if str(c).startswith("Passive Benchmark (")]
    rf_cols = [c for c in cols if str(c).startswith("Risk-Free Rate (")]

    if len(active_cols) == 0:
        raise KeyError("ret_ref_with_meta must contain a column starting with 'Active Benchmark ('")
    if len(rf_cols) == 0:
        raise KeyError("ret_ref_with_meta must contain a column starting with 'Risk-Free Rate ('")

    col_active = active_cols[0]
    col_rf = rf_cols[0]

    keep_ref_cols = active_cols + passive_cols + rf_cols
    keep_ref_cols = [c for c in keep_ref_cols if c in rr.columns]

    idx = cs.index.intersection(rr.index)
    if len(idx) == 0:
        raise ValueError("No overlapping dates between core_series and ret_ref_with_meta")

    out = pd.concat([cs.loc[idx].copy(), rr.loc[idx, keep_ref_cols].copy()], axis=1)

    out["ret_active_benchmark"] = out[col_active].astype(float).values
    out["ret_riskfree_rate"] = out[col_rf].astype(float).values

    out["value_active_benchmark"] = out["bench_bucket"].astype(float)
    out["value_riskfree_rate"] = out["rf_bucket"].astype(float)
    out["value_idle_cash"] = out["locked_cash"].astype(float)
    out["value_cash_total"] = out["value_active_benchmark"] + out["value_riskfree_rate"] + out["value_idle_cash"]
    out["value_stocks_plus_cash"] = out["stocks_plus_cash"].astype(float)

    # current sleeve balances, used downstream purely as ">0" masks (e.g. to hide a
    # sleeve's daily return on days it holds nothing)
    out["into_active_benchmark"] = out["value_active_benchmark"]
    out["into_riskfree_rate"] = out["value_riskfree_rate"]
    out["into_idle_cash"] = out["value_idle_cash"]

    if "bench_injection_today" in out.columns and "rf_injection_today" in out.columns:
        out["inj_active_benchmark"] = out["bench_injection_today"].astype(float)
        out["inj_riskfree_rate"] = out["rf_injection_today"].astype(float)
        out["inj_idle_cash"] = (
            out["cash_added_today"].astype(float)
            - out["inj_active_benchmark"]
            - out["inj_riskfree_rate"]
        )

    return out


def cut_strategy_df(strategy_df, t0):
    out = strategy_df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    t0 = pd.to_datetime(t0)
    if pd.isna(t0):
        return out.iloc[0:0].copy()

    return out.loc[out.index >= t0].copy()


def rank_range(a, b):
    a = int(a)
    b = int(b)
    if a <= 0 or b <= 0:
        raise ValueError("rank_range(a, b) requires a >= 1 and b >= 1")
    if b < a:
        raise ValueError(f"rank_range(a, b) requires b >= a, got a={a}, b={b}")
    return list(range(a, b + 1))


def momentum_filter(cum_1p, lookbacks, top_n, skip_t=0):
    lbs = list(lookbacks)
    tns_raw = list(top_n)

    if not (1 <= len(lbs) <= 3):
        raise ValueError("lookbacks must have length 1..3")
    if not (1 <= len(tns_raw) <= 3):
        raise ValueError("top_n must have length 1..3")
    if len(lbs) != len(tns_raw):
        raise ValueError("lookbacks and top_n must have the same length")

    tns = []
    for spec in tns_raw:
        if isinstance(spec, (list, tuple)):
            ranks = [int(r) for r in spec]
            if any(r <= 0 for r in ranks):
                raise ValueError("rank positions in top_n must be >= 1")
            if len(ranks) != len(set(ranks)):
                raise ValueError("rank positions in top_n must be unique")
            tns.append(ranks)
        else:
            tns.append(rank_range(1, spec))

    st = int(skip_t)
    if st < 0:
        raise ValueError("skip_t must be >= 0")

    df = cum_1p.copy()
    if "period_start" not in df.columns or "period_end" not in df.columns:
        raise KeyError("cum_1p must contain 'period_start' and 'period_end' columns")

    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.sort_values("period_end")

    signal_dates = pd.to_datetime(df["period_end"].values)

    def _txt(x):
        if pd.isna(x):
            return ""
        return str(x).strip().lower()

    day_map = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6,
    }

    def _next_weekday_after(d, target_wd):
        if pd.isna(d):
            return pd.NaT
        start = pd.Timestamp(d) + pd.Timedelta(days=1)
        add = (target_wd - start.weekday()) % 7
        out = start + pd.Timedelta(days=int(add))
        return pd.Timestamp(out.normalize())

    def _infer_exec(sd, freq_txt, exec_day_txt, hold_period):
        if pd.isna(sd):
            return pd.NaT

        h = 1 if pd.isna(hold_period) else int(max(1, hold_period))
        f = _txt(freq_txt)
        e = _txt(exec_day_txt)

        if e in ("month_start", "monthstart", "ms", "bms") or ("month" in f and "start" in e):
            rng = pd.date_range(start=pd.Timestamp(sd) + pd.Timedelta(days=1), periods=h, freq="BMS")
            return pd.Timestamp(rng[-1]) if len(rng) > 0 else pd.NaT

        if e in ("month_end", "monthend", "me", "bm") or ("month" in f and "end" in e):
            rng = pd.date_range(start=pd.Timestamp(sd) + pd.Timedelta(days=1), periods=h, freq="BM")
            return pd.Timestamp(rng[-1]) if len(rng) > 0 else pd.NaT

        wd = day_map.get(e, None)
        if wd is not None:
            first = _next_weekday_after(sd, wd)
            if pd.isna(first):
                return pd.NaT
            return pd.Timestamp(first + pd.Timedelta(days=7 * (h - 1)))

        if "month" in f:
            rng = pd.date_range(start=pd.Timestamp(sd) + pd.Timedelta(days=1), periods=h, freq="BM")
            return pd.Timestamp(rng[-1]) if len(rng) > 0 else pd.NaT

        return pd.NaT

    sig_anchor = pd.Series(signal_dates, index=df.index)
    if "signal_date" in df.columns:
        sig_anchor = pd.to_datetime(df["signal_date"], errors="coerce").combine_first(sig_anchor)
    if "actual_signal_date" in df.columns:
        sig_anchor = pd.to_datetime(df["actual_signal_date"], errors="coerce").combine_first(sig_anchor)

    actual_exec = pd.to_datetime(df["actual_execution_date"], errors="coerce") if "actual_execution_date" in df.columns else pd.Series(pd.NaT, index=df.index)
    planned_exec = pd.to_datetime(df["execution_date"], errors="coerce") if "execution_date" in df.columns else pd.Series(pd.NaT, index=df.index)
    assoc_actual_exec = pd.to_datetime(df["actual_execution_date_associated_signal_date"], errors="coerce") if "actual_execution_date_associated_signal_date" in df.columns else pd.Series(pd.NaT, index=df.index)
    assoc_planned_exec = pd.to_datetime(df["execution_date_associated_signal_date"], errors="coerce") if "execution_date_associated_signal_date" in df.columns else pd.Series(pd.NaT, index=df.index)
    fallback_exec = pd.to_datetime(df["period_start"].shift(-1), errors="coerce")

    freq_s = df["cfg_freq"] if "cfg_freq" in df.columns else pd.Series("weekly", index=df.index)
    exec_day_s = df["cfg_exec_day"] if "cfg_exec_day" in df.columns else pd.Series("Monday", index=df.index)
    hold_s = pd.to_numeric(df["cfg_hold_period"], errors="coerce") if "cfg_hold_period" in df.columns else pd.Series(1, index=df.index)

    inferred_exec = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for i in df.index:
        inferred_exec.loc[i] = _infer_exec(sig_anchor.loc[i], freq_s.loc[i], exec_day_s.loc[i], hold_s.loc[i])

    final_exec = (
        actual_exec
        .combine_first(planned_exec)
        .combine_first(assoc_actual_exec)
        .combine_first(assoc_planned_exec)
        .combine_first(inferred_exec)
        .combine_first(fallback_exec)
    )

    if "do_rebalance" in df.columns:
        trade_flag = df["do_rebalance"].fillna(False).astype(bool)
    elif "is_signal_day" in df.columns:
        trade_flag = df["is_signal_day"].fillna(False).astype(bool)
    elif "is_actual_signal_day" in df.columns:
        trade_flag = df["is_actual_signal_day"].fillna(False).astype(bool)
    else:
        trade_flag = pd.Series(True, index=df.index)

    use_mask = (trade_flag & final_exec.notna()).values
    exec_dates = pd.to_datetime(final_exec.values)

    meta_cols = []
    for c in df.columns:
        if c in ("period_start", "period_end"):
            meta_cols.append(c)
            continue
        if str(c).startswith("cfg_"):
            meta_cols.append(c)
            continue
        if c in (
            "weekday_num","weekday",
            "cal_is_month_first_obs","cal_is_month_last_obs",
            "is_signal_day","is_execution_day",
            "signal_date","execution_date",
            "execution_date_associated_signal_date",
            "signal_date_associated_with_execution_date",
            "is_actual_execution_day","is_actual_signal_day",
            "actual_signal_date","actual_execution_date",
            "actual_execution_date_associated_signal_date",
            "actual_signal_date_associated_with_execution_date",
            "is_rebalance_day","do_rebalance","rebalance_no",
        ):
            if c in df.columns:
                meta_cols.append(c)
            continue
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            meta_cols.append(c)

    meta_cols = list(dict.fromkeys(meta_cols))
    meta_base = df[meta_cols].copy()
    meta_base.index = signal_dates
    meta_base["signal_date"] = signal_dates
    meta_base["execution_date"] = exec_dates
    meta_base["is_effective_signal"] = use_mask

    tickers = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    if len(tickers) == 0:
        raise ValueError("No ticker columns found in cum_1p")

    onep = df[tickers].copy()
    onep.index = signal_dates
    onep_sig = onep.shift(st) if st > 0 else onep

    cumrets_map = {}
    ranks_map = {}
    n = len(signal_dates)

    for lb in lbs:
        L = int(lb)
        if L <= 0:
            raise ValueError("lookbacks must be >= 1")

        nan_in_window = onep_sig.isna().rolling(window=L, min_periods=1).sum() > 0
        cumprod = (1.0 + onep_sig).fillna(1.0).cumprod()
        window_prod = cumprod / cumprod.shift(L).fillna(1.0)
        cr = window_prod.where(~nan_in_window, np.nan) - 1.0

        cut = (L - 1) + st
        if cut > 0 and n > 0:
            cr.iloc[:cut, :] = np.nan

        rk = cr.rank(axis=1, ascending=False, method="min", na_option="bottom")
        if cut > 0 and n > 0:
            rk.iloc[:cut, :] = np.nan

        sd = pd.Series(signal_dates)
        sc_end = sd.shift(st)
        sc_start = sd.shift(st + (L - 1))

        if cut > 0 and n > 0:
            sc_end.iloc[:cut] = pd.NaT
            sc_start.iloc[:cut] = pd.NaT

        add_meta = pd.DataFrame(
            {
                "signal_creation_start_date": pd.to_datetime(sc_start.values),
                "signal_creation_end_date": pd.to_datetime(sc_end.values),
            },
            index=signal_dates,
        )

        cumrets_map[L] = pd.concat([meta_base, add_meta, cr], axis=1)
        ranks_map[L] = pd.concat([meta_base, add_meta, rk], axis=1)

    n_dates = len(signal_dates)
    m_tickers = len(tickers)
    tickers_arr = np.array(tickers, dtype=object)
    ticker_to_col = {t: j for j, t in enumerate(tickers)}

    tops_map = {}
    elig = np.ones((n_dates, m_tickers), dtype=bool)

    for s, (lb, ranks) in enumerate(zip(lbs, tns), start=1):
        L = int(lb)
        cols = [f"Top_{i}" for i in range(1, len(ranks) + 1)]

        arr = cumrets_map[L][tickers].to_numpy()
        masked = np.where(elig, arr, np.nan)
        valid_count = np.sum(~np.isnan(masked), axis=1)

        max_rank = max(ranks)
        effective = use_mask & (valid_count > 0)
        bad = effective & (valid_count < max_rank)
        if bad.any():
            first_bad = int(np.nonzero(bad)[0][0])
            raise ValueError(
                f"stage {s} (lookback={L}) requested rank {max_rank} on "
                f"{pd.Timestamp(signal_dates[first_bad]).date()}, but only "
                f"{int(valid_count[first_bad])} candidates were available that date"
            )

        order = np.argsort(-masked, axis=1, kind="stable")
        rank_idx = np.array([r - 1 for r in ranks])
        picked_cols = order[:, rank_idx]

        picks_arr = np.full((n_dates, len(ranks)), np.nan, dtype=object)
        if effective.any():
            picks_arr[effective, :] = tickers_arr[picked_cols[effective, :]]

        stage_df = pd.DataFrame(picks_arr, index=signal_dates, columns=cols, dtype=object)
        stage_df = pd.concat([meta_base, stage_df], axis=1)
        tops_map[f"stage{s}"] = stage_df

        flat = picks_arr.ravel()
        idx_map = pd.Series(flat).map(ticker_to_col)
        valid = idx_map.notna().to_numpy()
        row_idx = np.repeat(np.arange(n_dates), picks_arr.shape[1])[valid]
        col_idx = idx_map.to_numpy()[valid].astype(int)
        elig = np.zeros((n_dates, m_tickers), dtype=bool)
        elig[row_idx, col_idx] = True

    picks_df = tops_map[f"stage{len(lbs)}"].copy()

    exec_check = pd.DataFrame(index=signal_dates)
    exec_check["signal_anchor"] = pd.to_datetime(sig_anchor.values)
    exec_check["actual_execution_date"] = pd.to_datetime(actual_exec.values)
    exec_check["execution_date"] = pd.to_datetime(planned_exec.values)
    exec_check["actual_execution_date_associated_signal_date"] = pd.to_datetime(assoc_actual_exec.values)
    exec_check["execution_date_associated_signal_date"] = pd.to_datetime(assoc_planned_exec.values)
    exec_check["inferred_execution_date"] = pd.to_datetime(inferred_exec.values)
    exec_check["fallback_period_start_shift"] = pd.to_datetime(fallback_exec.values)
    exec_check["final_execution_date"] = pd.to_datetime(final_exec.values)
    exec_check["trade_flag"] = trade_flag.values
    exec_check["is_effective_signal"] = use_mask

    both = exec_check["actual_execution_date"].notna() & exec_check["inferred_execution_date"].notna()
    exec_check["inferred_matches_actual"] = np.where(
        both,
        exec_check["actual_execution_date"].values == exec_check["inferred_execution_date"].values,
        np.nan
    )

    return {
        "picks_df": picks_df,
        "tops_map": tops_map,
        "ranks_map": ranks_map,
        "cumrets_map": cumrets_map,
        "exec_check": exec_check,
        "lookbacks": [int(x) for x in lbs],
        "top_n": tns,
        "top_n_input": [int(x) if not isinstance(x, (list, tuple)) else list(x) for x in tns_raw],
    }


def ticker_performance_leaderboard(cum_1p, picks_df, top_cols=None):
    df = cum_1p.copy()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.sort_values("period_end")
    dates = pd.to_datetime(df["period_end"].values)

    tickers = [
        c for c in df.columns
        if c not in ("period_start", "period_end") and pd.api.types.is_numeric_dtype(df[c])
    ]
    R = df[tickers].copy()
    R.index = dates
    cp = (1.0 + R).cumprod()
    fwd = cp.shift(-1) / cp - 1.0

    if top_cols is None:
        top_cols = [c for c in picks_df.columns if str(c).startswith("Top_")]

    picks = picks_df.reindex(dates)[top_cols]

    rows = []
    for dt in dates:
        fwd_row = fwd.loc[dt]
        for col in top_cols:
            tkr = picks.loc[dt, col]
            if isinstance(tkr, str) and tkr:
                r = fwd_row.get(tkr, np.nan)
                if pd.notna(r):
                    rows.append((tkr, r))

    long_df = pd.DataFrame(rows, columns=["ticker", "fwd_ret"])
    summary = long_df.groupby("ticker")["fwd_ret"].agg(
        mean_fwd_ret="mean",
        median_fwd_ret="median",
        hit_rate=lambda s: (s > 0).mean(),
        n_obs="count",
    ).reset_index()

    return summary.sort_values("mean_fwd_ret", ascending=False).reset_index(drop=True)


def rank_forward_performance(cum_1p, ranks_map, horizons, tickers=None):
    df = cum_1p.copy()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.sort_values("period_end")
    dates = pd.to_datetime(df["period_end"].values)

    if tickers is None:
        tickers = [
            c for c in df.columns
            if c not in ("period_start", "period_end") and pd.api.types.is_numeric_dtype(df[c])
        ]

    R = df[tickers].copy()
    R.index = dates
    cp = (1.0 + R).cumprod()

    rows = []
    for L, rk_df in ranks_map.items():
        rk = rk_df.reindex(dates)[tickers]
        for h in horizons:
            h = int(h)
            fwd = cp.shift(-h) / cp - 1.0
            fwd = fwd.reindex(dates)

            joined = pd.DataFrame({
                "rank": rk.stack(),
                "fwd_ret": fwd.stack(),
            }).dropna()
            joined["rank"] = joined["rank"].astype(int)

            summary = joined.groupby("rank")["fwd_ret"].agg(
                mean_fwd_ret="mean",
                median_fwd_ret="median",
                hit_rate=lambda s: (s > 0).mean(),
                n_obs="count",
            ).reset_index()
            summary["lookback"] = L
            summary["horizon"] = h
            rows.append(summary)

    return pd.concat(rows, ignore_index=True)


def add_reference_cumret_columns(strategy_df_cut, ret_ref, suffix="_cumret"):
    s = strategy_df_cut.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    r = ret_ref.copy()
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()

    ref_cols = [c for c in r.columns if c in s.columns]
    if len(ref_cols) == 0:
        raise ValueError("No reference columns from ret_ref found in strategy_df_cut")

    out = s.copy()

    for c in ref_cols:
        x = pd.to_numeric(out[c], errors="coerce")
        if x.isna().any():
            bad = out.index[x.isna()][:5].tolist()
            raise ValueError(f"Column '{c}' has NaN values")
        out[f"{c}{suffix}"] = (1.0 + x).cumprod() - 1.0

    return out


def create_simple_returns_df(
    ret_slot,
    core_series,
    ref_returns,
    top_prefix="Top_",
    equity_return_col="equity_return",
    equity_value_col="equity_value_end",
    reference_cols=None,
    metadata_cols=None,
):
    rs = ret_slot.copy()
    cs = core_series.copy()
    rr = ref_returns.copy()

    rs.index = pd.to_datetime(rs.index, errors="coerce")
    cs.index = pd.to_datetime(cs.index, errors="coerce")
    rr.index = pd.to_datetime(rr.index, errors="coerce")

    rs = rs.loc[~rs.index.isna()].sort_index()
    cs = cs.loc[~cs.index.isna()].sort_index()
    rr = rr.loc[~rr.index.isna()].sort_index()

    top_cols = [c for c in rs.columns if str(c).startswith(top_prefix)]
    if len(top_cols) == 0:
        raise ValueError(f"No columns starting with '{top_prefix}' found in ret_slot")

    eq_ret_candidates = [equity_return_col, "equity_return", "ret_equity", "equity_ret", "stocks_return", "ret_stocks"]
    eq_ret_col = next((c for c in eq_ret_candidates if c in cs.columns), None)
    if eq_ret_col is None:
        raise KeyError(f"Could not find equity return column. Tried: {eq_ret_candidates}")

    eq_val_candidates = [equity_value_col, "equity_value_end", "equity_value", "stocks_value", "equity_value_start"]
    eq_val_col = next((c for c in eq_val_candidates if c in cs.columns), None)
    if eq_val_col is None:
        raise KeyError(f"Could not find equity value column. Tried: {eq_val_candidates}")

    if metadata_cols is None:
        meta_prefixes = ("cfg_", "weekday", "cal_", "is_", "signal", "execution", "actual")
        metadata_cols = [c for c in cs.columns if any(str(c).startswith(p) or (p in str(c)) for p in meta_prefixes)]
        metadata_cols = [c for c in metadata_cols if c not in {eq_ret_col, eq_val_col}]
        metadata_cols = list(dict.fromkeys(metadata_cols))

    if reference_cols is None:
        ref_tokens = ("ret_", "Benchmark", "Risk-Free", "Risk Free", "SPY", "BIL", "rf")
        reference_cols = [
            c for c in rr.columns
            if pd.api.types.is_numeric_dtype(rr[c]) and any(t in str(c) for t in ref_tokens)
        ]
        if len(reference_cols) == 0:
            meta_like = ("cfg_", "weekday", "cal_", "is_", "signal", "execution", "actual")
            reference_cols = [
                c for c in rr.columns
                if pd.api.types.is_numeric_dtype(rr[c]) and not any(m in str(c) for m in meta_like)
            ]

    if len(reference_cols) == 0:
        raise ValueError("No reference return columns found. Pass reference_cols explicitly.")

    common_idx = rs.index.intersection(cs.index).intersection(rr.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates across ret_slot, core_series, and ref_returns")

    meta_df = cs.loc[common_idx, [c for c in metadata_cols if c in cs.columns]].copy()
    tops_df = rs.loc[common_idx, top_cols].copy()
    eq_ret_df = cs.loc[common_idx, [eq_ret_col]].rename(columns={eq_ret_col: "ret_equity_sleeve"})
    eq_val_df = cs.loc[common_idx, [eq_val_col]].rename(columns={eq_val_col: "equity_value_sleeve"})
    ref_df = rr.loc[common_idx, [c for c in reference_cols if c in rr.columns]].copy()

    used = set(meta_df.columns) | set(tops_df.columns) | set(eq_ret_df.columns) | set(eq_val_df.columns)
    ref_df = ref_df.loc[:, [c for c in ref_df.columns if c not in used]]

    out = pd.concat([meta_df, tops_df, eq_ret_df, eq_val_df, ref_df], axis=1).sort_index()
    return out


def ret_to_overperformance_df(
    simple_ret_df,
    notional,
    agg="daily",
    benchmark_col=None,
    top_prefix="Top_",
    equity_ret_col="ret_equity_sleeve",
    equity_value_col="equity_value_sleeve",
    use_actual=True,
    keep_metadata=True,
):
    df = simple_ret_df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()

    top_cols = [c for c in df.columns if str(c).startswith(top_prefix)]
    if len(top_cols) == 0:
        raise ValueError(f"No columns starting with '{top_prefix}' found")

    if benchmark_col is None:
        passive_candidates = [c for c in df.columns if "Passive" in str(c)]
        if len(passive_candidates) == 1:
            benchmark_col = passive_candidates[0]
        elif len(passive_candidates) > 1:
            raise ValueError(f"Multiple passive benchmark columns found: {passive_candidates}. Pass benchmark_col explicitly.")
        else:
            fallback = [c for c in ["ret_passive_benchmark", "Passive Benchmark (SPY)", "ret_active_benchmark"] if c in df.columns]
            if len(fallback) == 0:
                raise KeyError("Could not auto-detect benchmark column. Pass benchmark_col explicitly.")
            benchmark_col = fallback[0]

    if equity_ret_col not in df.columns:
        raise KeyError(f"{equity_ret_col} not found")
    if equity_value_col not in df.columns:
        raise KeyError(f"{equity_value_col} not found")

    metadata_cols = [
        c for c in df.columns
        if (
            str(c).startswith("cfg_")
            or str(c).startswith("weekday")
            or str(c).startswith("cal_")
            or str(c).startswith("is_")
            or ("signal" in str(c))
            or ("execution" in str(c))
            or ("actual" in str(c))
        )
    ]

    ret_cols = top_cols + [benchmark_col, equity_ret_col]

    def _compound(s):
        x = pd.to_numeric(s, errors="coerce").dropna()
        if len(x) == 0:
            return np.nan
        return (1.0 + x).prod() - 1.0

    def _last_valid(s):
        x = pd.to_numeric(s, errors="coerce").dropna()
        if len(x) == 0:
            return np.nan
        return x.iloc[-1]

    def _end_meta(g):
        if not keep_metadata or len(metadata_cols) == 0:
            return {}
        m = g.iloc[-1][metadata_cols]
        return m.to_dict()

    if agg == "daily":
        out = df.copy()
        out["period_start"] = out.index
        out["period_end"] = out.index

    elif agg == "monthly":
        recs = []
        for _, g in df.groupby(df.index.to_period("M")):
            g = g.sort_index()
            start = g.index.min()
            end = g.index.max()

            rec = {"period_start": start, "period_end": end}
            for c in ret_cols:
                rec[c] = _compound(g[c])
            rec[equity_value_col] = _last_valid(g[equity_value_col])
            rec.update(_end_meta(g))
            recs.append(rec)

        out = pd.DataFrame(recs)
        if len(out) == 0:
            raise ValueError("No rows after monthly aggregation")
        out["period_start"] = pd.to_datetime(out["period_start"])
        out["period_end"] = pd.to_datetime(out["period_end"])
        out = out.set_index("period_end").sort_index()

    elif agg == "weekly":
        ex_col = None
        sg_col = None

        if use_actual and "is_actual_execution_day" in df.columns:
            ex_col = "is_actual_execution_day"
        elif "is_execution_day" in df.columns:
            ex_col = "is_execution_day"

        if use_actual and "is_actual_signal_day" in df.columns:
            sg_col = "is_actual_signal_day"
        elif "is_signal_day" in df.columns:
            sg_col = "is_signal_day"

        if ex_col is None:
            raise KeyError("Need is_actual_execution_day or is_execution_day for weekly aggregation")

        ex_flag = df[ex_col].fillna(False).astype(bool)
        if ex_flag.sum() == 0:
            raise ValueError("No execution days found for weekly aggregation")

        pid = ex_flag.cumsum().astype(float)
        pid[pid == 0] = np.nan

        recs = []
        valid = df.loc[pid.notna()].copy()
        pid_valid = pid.loc[pid.notna()]

        for _, g in valid.groupby(pid_valid):
            g = g.sort_index()
            start = g.index.min()

            if (sg_col is not None) and (sg_col in g.columns):
                sg_flag = g[sg_col].fillna(False).astype(bool)
                sig_idx = g.index[sg_flag]
                end = sig_idx.max() if len(sig_idx) > 0 else g.index.max()
            else:
                end = g.index.max()

            gw = g.loc[g.index <= end].copy()
            if len(gw) == 0:
                continue

            rec = {"period_start": start, "period_end": end}
            for c in ret_cols:
                rec[c] = _compound(gw[c])
            rec[equity_value_col] = _last_valid(gw[equity_value_col])
            rec.update(_end_meta(gw))
            recs.append(rec)

        out = pd.DataFrame(recs)
        if len(out) == 0:
            raise ValueError("No rows after weekly aggregation")
        out["period_start"] = pd.to_datetime(out["period_start"])
        out["period_end"] = pd.to_datetime(out["period_end"])
        out = out.set_index("period_end").sort_index()

    else:
        raise ValueError("agg must be one of: daily, weekly, monthly")

    for c in ret_cols + [equity_value_col]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    comp = out[top_cols].sub(out[benchmark_col], axis=0)
    out["n_overperformers"] = comp.gt(0).sum(axis=1)
    out["n_underperformers"] = comp.lt(0).sum(axis=1)
    out["n_compared"] = comp.notna().sum(axis=1)
    out["equity_excess_return"] = out[equity_ret_col] - out[benchmark_col]
    out["equity_below_notional"] = (float(notional) - out[equity_value_col]).clip(lower=0.0)

    keep_cols = ["period_start"]
    if keep_metadata and len(metadata_cols) > 0:
        keep_cols += [c for c in metadata_cols if c in out.columns]
    keep_cols += top_cols + [benchmark_col, equity_ret_col, equity_value_col]
    keep_cols += ["n_overperformers", "n_underperformers", "n_compared", "equity_excess_return", "equity_below_notional"]
    keep_cols = [c for c in keep_cols if c in out.columns]

    out = out[keep_cols]
    return out


def strategy_return_from_weight_and_return_df(weights_df, returns_df):
    w = weights_df.copy().dropna()
    r = returns_df.copy().dropna()

    w.index = pd.to_datetime(w.index, errors="coerce")
    r.index = pd.to_datetime(r.index, errors="coerce")

    w = w.loc[w.index.notna()]
    r = r.loc[r.index.notna()]

    idx = w.index.intersection(r.index)
    cols = w.columns.intersection(r.columns)

    if len(idx) == 0:
        raise ValueError("No common index")
    if len(cols) == 0:
        raise ValueError("No common columns")

    s = (
        w.loc[idx, cols].apply(pd.to_numeric, errors="coerce")
        * r.loc[idx, cols].apply(pd.to_numeric, errors="coerce")
    ).sum(axis=1)

    s.name = "Equity Sleeve"
    return s


def build_heatmap_panel(
    simple_ret_df,
    equity_sleeve_ret,
    strategy_df_cut=None,
    strategy_ret=None,
    strategy_value_col="value_stocks_plus_cash",
    top_prefix="Top_",
    passive_keyword="Passive",
):
    s = simple_ret_df.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[s.index.notna()].sort_index()
    s = s.loc[~s.index.duplicated(keep="last")]

    eq = equity_sleeve_ret.copy()
    if isinstance(eq, pd.DataFrame):
        if eq.shape[1] != 1:
            raise ValueError("equity_sleeve_ret DataFrame must have exactly one column")
        eq = eq.iloc[:, 0]
    eq.index = pd.to_datetime(eq.index, errors="coerce")
    eq = eq.loc[eq.index.notna()].sort_index()
    eq = eq.loc[~eq.index.duplicated(keep="last")]
    eq = pd.to_numeric(eq, errors="coerce")
    eq.name = "Equity Sleeve"

    if strategy_ret is not None:
        st = strategy_ret.copy()
        if isinstance(st, pd.DataFrame):
            if st.shape[1] != 1:
                raise ValueError("strategy_ret DataFrame must have exactly one column")
            st = st.iloc[:, 0]
        st.index = pd.to_datetime(st.index, errors="coerce")
        st = st.loc[st.index.notna()].sort_index()
        st = st.loc[~st.index.duplicated(keep="last")]
        st = pd.to_numeric(st, errors="coerce")
        st.name = "Strategy"
    else:
        if strategy_df_cut is None:
            raise ValueError("Provide either strategy_ret or strategy_df_cut")
        t = strategy_df_cut.copy()
        t.index = pd.to_datetime(t.index, errors="coerce")
        t = t.loc[t.index.notna()].sort_index()
        t = t.loc[~t.index.duplicated(keep="last")]
        if strategy_value_col not in t.columns:
            raise KeyError(f"Missing in strategy_df_cut: {strategy_value_col}")
        st = pd.to_numeric(t[strategy_value_col], errors="coerce").pct_change()
        st.name = "Strategy"

    kw = str(passive_keyword).lower().strip()
    passive_candidates = [c for c in s.columns if kw in str(c).lower()]
    if len(passive_candidates) == 0:
        raise KeyError(f"No column in simple_ret_df contains '{passive_keyword}'")

    def _score(c):
        z = str(c).lower()
        sc = 0
        if "passive" in z:
            sc += 10
        if "benchmark" in z:
            sc += 5
        return sc

    passive_col = sorted(passive_candidates, key=lambda x: (-_score(x), str(x)))[0]

    top_cols = [c for c in s.columns if str(c).startswith(top_prefix)]

    def _top_num(c):
        try:
            return int(str(c).split("_")[1])
        except Exception:
            return 10**9

    top_cols = sorted(top_cols, key=_top_num)

    idx = s.index.intersection(eq.index).intersection(st.index).sort_values()
    if len(idx) == 0:
        raise ValueError("No common index across simple_ret_df, equity_sleeve_ret, strategy source")

    s2 = s.loc[idx]
    eq2 = eq.loc[idx]
    st2 = st.loc[idx]

    out = pd.DataFrame(index=idx)

    for c in top_cols:
        out[c] = pd.to_numeric(s2[c], errors="coerce")

    out["Equity Sleeve"] = eq2
    out["Strategy"] = st2
    out[passive_col] = pd.to_numeric(s2[passive_col], errors="coerce")

    series_order = top_cols + ["Equity Sleeve", "Strategy", passive_col]
    out = out.loc[:, series_order]

    return {
        "returns_panel": out,
        "series_order": series_order,
        "top_cols": top_cols,
        "passive_col": passive_col,
        "common_index": idx,
    }


def aggregate_heatmap_panel(returns_panel, by="month", include_weekends=False):
    d = returns_panel.copy()
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d.loc[d.index.notna()].sort_index()

    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    mode = str(by).lower().strip()
    if mode not in ("month", "weekday"):
        raise ValueError("by must be 'month' or 'weekday'")

    if mode == "month":
        group_name = "month"
        vals = d.index.month_name().str[:3]
        order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    else:
        group_name = "weekday"
        vals = d.index.day_name().str[:3]
        order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"] if include_weekends else ["Mon","Tue","Wed","Thu","Fri"]

    d["__grp__"] = pd.Categorical(vals, categories=order, ordered=True)
    d = d.loc[d["__grp__"].notna()]

    long_df = d.reset_index(drop=True).melt(
        id_vars="__grp__",
        value_vars=[c for c in d.columns if c != "__grp__"],
        var_name="series",
        value_name="value"
    )

    agg_long = (
        long_df.groupby(["__grp__", "series"], observed=False)["value"]
        .agg(mean="mean", count="count")
        .reset_index()
        .rename(columns={"__grp__": group_name})
    )

    idx = pd.Index(order, name=group_name)
    series_order = [c for c in d.columns if c != "__grp__"]

    mean_table = agg_long.pivot(index=group_name, columns="series", values="mean").reindex(index=idx, columns=series_order)
    count_table = agg_long.pivot(index=group_name, columns="series", values="count").reindex(index=idx, columns=series_order).fillna(0).astype(int)

    return {
        "by": mode,
        "group_name": group_name,
        "group_order": order,
        "series_order": series_order,
        "mean_table": mean_table,
        "count_table": count_table,
        "agg_long": agg_long,
    }


def merge_ret_slot_with_metadata_from_t0(ret_slot, metadata, t0, how="inner"):
    r = ret_slot.copy()
    if isinstance(r, pd.Series):
        r = r.to_frame(name=r.name if r.name is not None else "ret_slot")

    m = metadata.copy()

    r.index = pd.to_datetime(r.index, errors="coerce")
    r = r.loc[r.index.notna()]
    r = r.loc[~r.index.duplicated(keep="last")]

    if "Date" in m.columns:
        m = m.copy()
        m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
        m = m.set_index("Date")
    else:
        m.index = pd.to_datetime(m.index, errors="coerce")

    m = m.loc[m.index.notna()]
    m = m.loc[~m.index.duplicated(keep="last")]

    out = r.join(m, how=how)
    out = out.loc[out.index >= pd.Timestamp(t0)].sort_index()
    return out


def cumulative_top_histories(
    ret_slot_with_meta,
    top_prefix="Top_",
    exec_col="is_actual_execution_day",
    signal_col="is_actual_signal_day",
    include_exec_day=True,
    include_signal_day=True,
    close_open_spell_at_last_date=True,
    fillna_ret=0.0,
):
    d = ret_slot_with_meta.copy()
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d.loc[d.index.notna()].sort_index()
    d = d.loc[~d.index.duplicated(keep="last")]

    if exec_col not in d.columns:
        raise KeyError(f"Missing column: {exec_col}")
    if signal_col not in d.columns:
        raise KeyError(f"Missing column: {signal_col}")

    top_cols = [c for c in d.columns if str(c).startswith(top_prefix)]
    if len(top_cols) == 0:
        raise ValueError(f"No columns starting with {top_prefix}")

    def _k(x):
        try:
            return int(str(x).split("_")[1])
        except Exception:
            return 10**9

    top_cols = sorted(top_cols, key=_k)

    exec_dates = list(d.index[d[exec_col].fillna(False).astype(bool)])
    signal_dates = list(d.index[d[signal_col].fillna(False).astype(bool)])
    idx = d.index

    spells = []
    j = 0
    spell_id = 0

    for e in exec_dates:
        while j < len(signal_dates) and signal_dates[j] < e:
            j += 1

        if j < len(signal_dates):
            s = signal_dates[j]
            j += 1
        else:
            if not close_open_spell_at_last_date:
                continue
            s = idx[-1]

        i0 = idx.get_loc(e)
        i1 = idx.get_loc(s)

        if not include_exec_day:
            i0 += 1
        if not include_signal_day:
            i1 -= 1

        if i0 > i1 or i0 >= len(idx) or i1 < 0:
            continue

        spell_id += 1
        spells.append(
            {
                "spell_id": spell_id,
                "exec_date": e,
                "signal_date": s,
                "start_date": idx[i0],
                "end_date": idx[i1],
                "start_i": int(i0),
                "end_i": int(i1),
                "n_days": int(i1 - i0 + 1),
            }
        )

    spells_df = pd.DataFrame(spells)

    if len(spells_df) == 0:
        return {
            "spells": spells_df,
            "cum_long": pd.DataFrame(columns=["Date", "spell_id", "slot", "ret", "cum"]),
            "cum_wide": pd.DataFrame(),
        }

    rows = []

    for _, sp in spells_df.iterrows():
        i0, i1, sid = int(sp["start_i"]), int(sp["end_i"]), int(sp["spell_id"])
        win_idx = idx[i0:i1 + 1]

        for c in top_cols:
            r = pd.to_numeric(d.loc[win_idx, c], errors="coerce")
            if fillna_ret is not None:
                r = r.fillna(float(fillna_ret))

            if len(r) == 0:
                continue

            cum = (1.0 + r).cumprod()
            if pd.notna(cum.iloc[0]) and cum.iloc[0] != 0:
                cum = cum / cum.iloc[0]
            if len(cum) > 0 and pd.notna(cum.iloc[0]):
                cum.iloc[0] = 1.0

            tmp = pd.DataFrame(
                {
                    "Date": win_idx,
                    "spell_id": sid,
                    "slot": c,
                    "ret": r.values,
                    "cum": cum.values,
                }
            )
            rows.append(tmp)

    cum_long = pd.concat(rows, ignore_index=True) if len(rows) else pd.DataFrame(columns=["Date", "spell_id", "slot", "ret", "cum"])
    cum_wide = cum_long.pivot(index="Date", columns=["slot", "spell_id"], values="cum").sort_index()

    return {
        "spells": spells_df.drop(columns=["start_i", "end_i"]),
        "cum_long": cum_long,
        "cum_wide": cum_wide,
    }


def build_default_history_stats(
    hist,
    slot_col="slot",
    spell_col="spell_id",
    cum_col="cum",
    top_prefix="Top_",
    min_obs=1,
):
    d = hist["cum_long"].copy() if isinstance(hist, dict) else hist.copy()

    d = d[[slot_col, spell_col, cum_col]].copy()
    d[slot_col] = d[slot_col].astype(str)
    d = d[d[slot_col].str.startswith(top_prefix)]
    d[cum_col] = pd.to_numeric(d[cum_col], errors="coerce")
    d = d.dropna(subset=[cum_col, slot_col, spell_col]).copy()

    d = d.sort_values([slot_col, spell_col]).copy()
    d["h"] = d.groupby([slot_col, spell_col]).cumcount()

    first_vals = d.groupby([slot_col, spell_col])[cum_col].transform("first")
    d["cum_norm"] = d[cum_col] / first_vals
    d.loc[d["h"] == 0, "cum_norm"] = 1.0

    stats = (
        d.groupby([slot_col, "h"], observed=False)["cum_norm"]
        .agg(
            mean="mean",
            std="std",
            p10=lambda x: np.nanpercentile(x, 10),
            p90=lambda x: np.nanpercentile(x, 90),
            n="count",
        )
        .reset_index()
    )

    stats["std"] = stats["std"].fillna(0.0)
    stats["mean_plus_std"] = stats["mean"] + stats["std"]
    stats["mean_minus_std"] = stats["mean"] - stats["std"]
    stats = stats.loc[stats["n"] >= int(min_obs)].copy()

    def _k(x):
        try:
            return int(str(x).split("_")[1])
        except Exception:
            return 10**9

    slot_order = sorted(stats[slot_col].dropna().unique().tolist(), key=_k)

    return {
        "stats": stats,
        "slot_order": slot_order,
        "slot_col": slot_col,
    }


def check_cash_invariant(
    dt, equity_start, pre_locked, pre_bench, pre_rf, initial_injection,
    pos_sum, locked_cash, bench_bucket, rf_bucket,
    trade_sell_cost, trade_buy_cost, cash_tol=1e-6,
):
    """Raise if post-trade cash + positions don't match pre-trade cash + positions + costs.

    This is the ledger's primary safety net against double-counting or phantom cash;
    every execution day's trade block must balance to within cash_tol.
    """
    pre_total = float(equity_start) + float(pre_locked) + float(pre_bench) + float(pre_rf) + float(initial_injection)
    post_total = (
        float(pos_sum) + float(locked_cash) + float(bench_bucket) + float(rf_bucket)
        + float(trade_sell_cost) + float(trade_buy_cost)
    )
    gap = post_total - pre_total
    if abs(gap) > float(cash_tol):
        raise ValueError(
            f"Cash accounting mismatch on {pd.Timestamp(dt).date()}: "
            f"gap={gap:.6g}. (This indicates double-counting or phantom cash.)"
        )
    return gap


def simulate_portfolio_ledger(
    ret_ac,
    metadata,
    holdings_by_slot_daily,
    notional,
    top_k,
    slippage_bps=0.0,
    winner_handling="let_ride",
    loser_handling="leave_alone",
    entry_funding="capped_entry",
    position_sizing="equal",
    inv_vol_weights_exec=None,
    cash_reinvest_frac=1.0,
    cash_alloc_bench=0.0,
    cash_alloc_rf=0.0,
    bench_ret=None,
    rf_ret=None,
    entry_ramp_frac=1.0,
    check_cash=True,
    cash_tol=1e-6,
):
    r = ret_ac.copy()
    r.index = pd.to_datetime(r.index, errors="coerce")
    r = r.loc[r.index.notna()].sort_index()

    md = metadata.copy()
    md.index = pd.to_datetime(md.index, errors="coerce")
    md = md.loc[md.index.notna()].sort_index()

    hs = holdings_by_slot_daily.copy()
    hs.index = pd.to_datetime(hs.index, errors="coerce")
    hs = hs.loc[hs.index.notna()].sort_index()

    K = int(top_k)
    if K <= 0:
        raise ValueError("top_k must be >= 1")

    slip_bps = float(slippage_bps)
    slip = slip_bps / 10000.0
    if slip < 0:
        raise ValueError("slippage_bps must be >= 0")

    winner_handling = str(winner_handling).strip().lower()
    if winner_handling not in ("let_ride", "trim_to_target"):
        raise ValueError("winner_handling must be 'let_ride' or 'trim_to_target'")

    loser_handling = str(loser_handling).strip().lower()
    if loser_handling not in ("leave_alone", "top_up_to_target"):
        raise ValueError("loser_handling must be 'leave_alone' or 'top_up_to_target'")

    entry_funding = str(entry_funding).strip().lower()
    if entry_funding not in ("full_target", "capped_entry"):
        raise ValueError("entry_funding must be 'full_target' or 'capped_entry'")
    bm = "proceeds_prorata" if entry_funding == "full_target" else "capped_ticket"

    position_sizing = str(position_sizing).strip().lower()
    if position_sizing in ("equal_weighting", "equal", "ew", "equalweight", "equal_weight"):
        wt = "equal"
    elif position_sizing in ("inverse_vol", "inv_vol_weighting", "inv_vol", "invvol", "inverse_volatility", "inv_volatility"):
        wt = "inv_vol"
    else:
        raise ValueError("position_sizing must be 'equal' or 'inverse_vol'")

    exec_flag = "is_actual_execution_day" if "is_actual_execution_day" in md.columns else "is_execution_day"
    if exec_flag not in md.columns:
        raise KeyError("metadata must contain is_actual_execution_day or is_execution_day")

    sig_col = "actual_signal_date" if "actual_signal_date" in md.columns else "signal_date"
    if sig_col not in md.columns:
        raise KeyError("metadata must contain actual_signal_date or signal_date")

    slot_cols = [c for c in hs.columns if str(c).startswith("Top_")]
    if len(slot_cols) == 0:
        raise ValueError("holdings_by_slot_daily must contain Top_* columns")

    def _slot_key(c):
        s = str(c)
        n = s.split("_", 1)[1] if "_" in s else ""
        try:
            return int(n)
        except Exception:
            return 10**9

    slot_cols = sorted(slot_cols, key=_slot_key)[:K]
    if len(slot_cols) < K:
        raise ValueError(f"holdings_by_slot_daily has only {len(slot_cols)} Top_* columns but top_k={K}")

    idx = md.index.intersection(r.index).intersection(hs.index)
    idx = pd.DatetimeIndex(idx).sort_values()
    if len(idx) == 0:
        raise ValueError("No overlapping dates among ret_ac, metadata, holdings_by_slot_daily")

    md = md.loc[idx]
    r = r.loc[idx]
    hs = hs.loc[idx]

    tickers = list(r.columns)
    if len(tickers) == 0:
        raise ValueError("ret_ac has no ticker columns")

    colpos = {t: j for j, t in enumerate(tickers)}

    wexec = None
    if wt == "inv_vol":
        if inv_vol_weights_exec is None:
            raise ValueError("inv_vol_weights_exec must be provided when weighting='inv_vol'")
        wexec = inv_vol_weights_exec.copy()
        wexec.index = pd.to_datetime(wexec.index, errors="coerce")
        wexec = wexec.loc[wexec.index.notna()].sort_index()
        wcols = [c for c in wexec.columns if c in colpos]
        if len(wcols) == 0:
            raise ValueError("inv_vol_weights_exec has no ticker columns that match ret_ac columns")
        wexec = wexec[wcols]

    n = len(idx)
    m = len(tickers)

    ret_arr = r.to_numpy()
    value_end = np.zeros((n, m), dtype=float)
    weight_start = np.zeros((n, m), dtype=float)

    core = md.copy()
    core["locked_cash"] = 0.0
    core["bench_bucket"] = 0.0
    core["rf_bucket"] = 0.0
    core["cash_added_today"] = 0.0
    core["bench_injection_today"] = 0.0
    core["rf_injection_today"] = 0.0
    core["current_notional"] = float(notional)
    core["equity_value_start"] = 0.0
    core["equity_value_end"] = 0.0
    core["equity_return"] = 0.0
    core["stocks_plus_cash"] = 0.0

    exec_mask = md[exec_flag].astype(bool).to_numpy()
    sig_vals = pd.to_datetime(md[sig_col]).to_numpy()

    top1_vals = hs[slot_cols[0]].to_numpy(dtype=object)
    valid_top1 = np.array([(isinstance(x, str) and x) for x in top1_vals], dtype=bool)

    if float(notional) < 0:
        raise ValueError("notional must be >= 0")

    reinvest_frac = float(cash_reinvest_frac)
    if not (0.0 <= reinvest_frac <= 1.0):
        raise ValueError("cash_reinvest_frac must be between 0.0 and 1.0")

    entry_ramp = float(entry_ramp_frac)
    if not (0.0 < entry_ramp <= 1.0):
        raise ValueError("entry_ramp_frac must be in (0.0, 1.0]")

    cash_alloc_bench = float(cash_alloc_bench)
    cash_alloc_rf = float(cash_alloc_rf)
    validate_cash_split(reinvest_frac, cash_alloc_bench, cash_alloc_rf)

    if cash_alloc_bench > 0.0 and bench_ret is None:
        raise ValueError("bench_ret must be provided when cash_alloc_bench > 0")
    if cash_alloc_rf > 0.0 and rf_ret is None:
        raise ValueError("rf_ret must be provided when cash_alloc_rf > 0")

    bench_ret_arr = np.zeros(n, dtype=float)
    if bench_ret is not None:
        br = bench_ret.copy()
        br.index = pd.to_datetime(br.index, errors="coerce")
        br = br.loc[br.index.notna()].sort_index()
        bench_ret_arr = pd.to_numeric(br.reindex(idx), errors="coerce").fillna(0.0).to_numpy()

    rf_ret_arr = np.zeros(n, dtype=float)
    if rf_ret is not None:
        rr_ = rf_ret.copy()
        rr_.index = pd.to_datetime(rr_.index, errors="coerce")
        rr_ = rr_.loc[rr_.index.notna()].sort_index()
        rf_ret_arr = pd.to_numeric(rr_.reindex(idx), errors="coerce").fillna(0.0).to_numpy()

    def _weights_for_date(dt, desired_list):
        if len(desired_list) == 0:
            return pd.Series(dtype=float)
        if wt == "equal":
            return pd.Series(1.0 / float(len(desired_list)), index=desired_list, dtype=float)

        if wexec is None:
            return pd.Series(1.0 / float(len(desired_list)), index=desired_list, dtype=float)

        dtt = pd.Timestamp(dt)
        if dtt not in wexec.index:
            return pd.Series(1.0 / float(len(desired_list)), index=desired_list, dtype=float)

        row = wexec.loc[dtt]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        w = pd.to_numeric(row, errors="coerce")
        w2 = w.reindex(desired_list).astype(float)

        if w2.isna().any() or (w2 < 0).any():
            return pd.Series(1.0 / float(len(desired_list)), index=desired_list, dtype=float)

        s = float(w2.sum())
        if (not np.isfinite(s)) or s <= 0:
            return pd.Series(1.0 / float(len(desired_list)), index=desired_list, dtype=float)

        return w2 / s

    exec_rows = []
    initialized = False
    locked_cash = 0.0
    bench_bucket = 0.0
    rf_bucket = 0.0
    current_notional = float(notional)
    pos = np.zeros(m, dtype=float)

    for i, dt in enumerate(idx):
        cash_added_today = 0.0
        bench_injection_today = 0.0
        rf_injection_today = 0.0
        equity_start = float(pos.sum())
        core.iloc[i, core.columns.get_loc("equity_value_start")] = equity_start

        trade_sell_cost = 0.0
        trade_buy_cost = 0.0
        gross_sell = 0.0
        gross_buy = 0.0
        n_sells = 0
        n_trims = 0
        n_buys = 0
        buy_need = np.nan
        unfilled_need = np.nan
        buy_allocated = 0.0
        inv_vol_used = (wt == "inv_vol")

        pre_locked = float(locked_cash)
        pre_bench = float(bench_bucket)
        pre_rf = float(rf_bucket)

        if exec_mask[i] and valid_top1[i]:
            row = hs.loc[dt, slot_cols].tolist()
            desired_list = []
            seen = set()
            for x in row:
                if isinstance(x, str) and x:
                    if x in seen:
                        continue
                    seen.add(x)
                    desired_list.append(x)

            if len(desired_list) > 0:
                desired_set = set(desired_list)

                held_idx = np.flatnonzero(pos > 0.0)
                held_set = set([tickers[j] for j in held_idx])

                initial_injection = 0.0
                if (not initialized) and equity_start == 0.0 and float(notional) > 0.0:
                    initial_injection = float(notional)
                    initialized = True

                # pre_locked (this week's starting circulating reinvest-cash balance) counts
                # toward the investable pool - unlike bench_bucket/rf_bucket, this money is
                # earmarked for stocks, just temporarily undeployed, so it belongs in base_pool
                # and is available as buying power below (see net_cash_available).
                base_pool = float(equity_start) + float(pre_locked) + float(initial_injection)
                profit_above = max(0.0, base_pool - float(current_notional))
                pool_today = min(base_pool, float(current_notional)) + reinvest_frac * profit_above
                ticket_today = pool_today / float(K)

                bench_injection_target = cash_alloc_bench * profit_above
                rf_injection_target = cash_alloc_rf * profit_above

                leavers = held_set - desired_set
                sell_leavers = 0.0
                if len(leavers) > 0:
                    for t in leavers:
                        j = colpos.get(t, None)
                        if j is None:
                            raise ValueError(f"Held ticker not found in ret_ac columns: {t}")
                        v = float(pos[j])
                        if v > 0.0:
                            sell_leavers += v
                            n_sells += 1
                            pos[j] = 0.0

                # Unified target-weight path for both position_sizing modes: _weights_for_date
                # already returns uniform 1/len(desired_list) weights when wt=="equal", so equal
                # weighting is just the uniform-weight special case of the inv_vol math below
                # (proven behavior-identical to the old separate equal-weight branch by
                # test_equal_matches_inv_vol_with_uniform_weights in test_momentum_backend.py).
                w_target = _weights_for_date(dt, desired_list)
                target_dollars = (w_target * float(pool_today)).to_dict()

                # Trim step: winners (re-selected, currently above target) are trimmed only
                # under winner_handling=="trim_to_target"; under "let_ride" they're left to
                # compound uncapped until they fully leave the selection (handled by `leavers`).
                sell_trims = 0.0
                if winner_handling == "trim_to_target":
                    for t in desired_list:
                        j = colpos.get(t, None)
                        if j is None:
                            raise ValueError(f"Desired ticker not found in ret_ac columns: {t}")
                        cur = float(pos[j])
                        tgt = float(target_dollars.get(t, 0.0))
                        if cur > tgt + 1e-12:
                            trim = cur - tgt
                            sell_trims += trim
                            n_trims += 1
                            n_sells += 1
                            pos[j] = tgt

                gross_sell = float(sell_leavers + sell_trims)
                trade_sell_cost = gross_sell * slip
                net_cash_from_sells = gross_sell * (1.0 - slip) if gross_sell > 0 else 0.0
                net_cash_available = net_cash_from_sells + float(initial_injection) + float(pre_locked)
                gross_buy_capacity = net_cash_available / (1.0 + slip) if slip > 0 else net_cash_available

                # Funding-eligibility step: new entrants (cur<=0) are always eligible; a
                # re-selected loser (cur>0 but below target) is eligible only under
                # loser_handling=="top_up_to_target" - independent of winner_handling.
                # entry_ramp_frac only caps true entrants, never an established loser topping
                # back up (it's not a new position, so there's no "ramp" to speak of).
                deficits = {}
                entrants = set()
                for t in desired_list:
                    j = colpos[t]
                    cur = float(pos[j])
                    tgt = float(target_dollars.get(t, 0.0))
                    d = max(0.0, tgt - cur)
                    if d <= 0.0:
                        continue
                    if cur <= 0.0:
                        deficits[t] = d
                        entrants.add(t)
                    elif loser_handling == "top_up_to_target":
                        deficits[t] = d

                if len(deficits) == 0:
                    cash_added_today = net_cash_available
                    n_buys = 0
                    if bm == "capped_ticket":
                        buy_need = 0.0
                        unfilled_need = 0.0
                        buy_allocated = 0.0
                else:
                    eff_def = deficits.copy()
                    if bm == "capped_ticket" and len(entrants) > 0:
                        for t in entrants:
                            entry_cap = entry_ramp * float(target_dollars.get(t, 0.0))
                            eff_def[t] = min(float(eff_def[t]), entry_cap)

                    total_eff = float(sum(eff_def.values()))
                    if total_eff <= 0:
                        cash_added_today = net_cash_available
                        n_buys = 0
                        if bm == "capped_ticket":
                            buy_need = 0.0
                            unfilled_need = 0.0
                            buy_allocated = 0.0
                    else:
                        if bm == "capped_ticket":
                            buy_need = float(total_eff)
                            spend = min(float(gross_buy_capacity), float(total_eff))
                            unfilled_need = max(0.0, float(total_eff) - float(spend))
                        else:
                            # full_target/proceeds_prorata: deploy all available capacity,
                            # prorated by deficit share, not capped at the nominal target.
                            spend = float(gross_buy_capacity)

                        if spend > 0.0:
                            for t, d in eff_def.items():
                                add = spend * (float(d) / float(total_eff)) if total_eff > 0 else 0.0
                                pos[colpos[t]] += float(add)
                            gross_buy = float(spend)

                        trade_buy_cost = gross_buy * slip
                        cash_added_today = net_cash_available - (gross_buy + trade_buy_cost)
                        if abs(cash_added_today) < 1e-9:
                            cash_added_today = 0.0
                        if cash_added_today < -1e-8:
                            raise ValueError("Buy allocation exceeded available cash (numerical issue)")

                        n_buys = int(sum(1 for t in eff_def if eff_def[t] > 0.0 and (pos[colpos[t]] > 0.0)))
                        buy_allocated = gross_buy

                bench_injection_today = min(bench_injection_target, max(0.0, cash_added_today))
                rf_injection_today = min(rf_injection_target, max(0.0, cash_added_today - bench_injection_today))
                cash_added_to_locked = cash_added_today - bench_injection_today - rf_injection_today

                # locked_cash is now a circulating reinvest-cash balance, not a dead-end: it's
                # REPLACED each exec day (not accumulated), because pre_locked was already spent
                # into net_cash_available/gross_buy_capacity above - whatever's left after that
                # full day's flow (buys + bench/rf skim) IS the new balance in its entirety.
                locked_cash = float(cash_added_to_locked)
                bench_bucket += float(bench_injection_today)
                rf_bucket += float(rf_injection_today)

                # Ratchet: the reinvested share of today's gain permanently raises the notional
                # watermark, so skimming to bench/rf only ever resumes above the highest level
                # ever reached - a past reinvested gain is never re-skimmed on a later pullback.
                current_notional += reinvest_frac * profit_above

                if bool(check_cash):
                    check_cash_invariant(
                        dt, equity_start, pre_locked, pre_bench, pre_rf, initial_injection,
                        pos.sum(), locked_cash, bench_bucket, rf_bucket,
                        trade_sell_cost, trade_buy_cost, cash_tol,
                    )

                exec_rows.append(
                    {
                        "execution_date": dt,
                        "signal_date_associated": pd.to_datetime(sig_vals[i]),
                        "initial_injection": float(initial_injection),
                        "n_sells": int(n_sells),
                        "n_trims": int(n_trims),
                        "n_buys": int(n_buys),
                        "gross_proceeds": float(gross_sell),
                        "dollars_traded_sell": float(gross_sell),
                        "dollars_traded_buy": float(gross_buy),
                        "sell_cost_dollars": float(trade_sell_cost),
                        "buy_cost_dollars": float(trade_buy_cost),
                        "total_cost_dollars": float(trade_sell_cost + trade_buy_cost),
                        "net_cash_available": float(net_cash_available),
                        "gross_buy_capacity": float(gross_buy_capacity),
                        "buy_need": float(buy_need) if bm == "capped_ticket" else np.nan,
                        "buy_allocated": float(buy_allocated) if bm == "capped_ticket" else float(gross_buy),
                        "unfilled_buy_need": float(unfilled_need) if bm == "capped_ticket" else np.nan,
                        "locked_cash_added": float(cash_added_to_locked),
                        "bench_injection": float(bench_injection_today),
                        "rf_injection": float(rf_injection_today),
                        "profit_above_notional": float(profit_above),
                        "winner_handling": winner_handling,
                        "loser_handling": loser_handling,
                        "entry_funding": entry_funding,
                        "position_sizing": wt,
                        "cash_reinvest_frac": float(reinvest_frac),
                        "entry_ramp_frac": float(entry_ramp),
                        "pool": float(pool_today),
                        "current_notional": float(current_notional),
                        "inv_vol_used": bool(inv_vol_used),
                        "slippage_bps": float(slip_bps),
                        "ticket": float(ticket_today),
                    }
                )

        core.iloc[i, core.columns.get_loc("cash_added_today")] = float(cash_added_today)
        core.iloc[i, core.columns.get_loc("bench_injection_today")] = float(bench_injection_today)
        core.iloc[i, core.columns.get_loc("rf_injection_today")] = float(rf_injection_today)
        core.iloc[i, core.columns.get_loc("locked_cash")] = float(locked_cash)
        core.iloc[i, core.columns.get_loc("current_notional")] = float(current_notional)

        eq_start_after_trades = float(pos.sum())
        w = np.zeros(m, dtype=float)
        if eq_start_after_trades > 0.0:
            w = pos / eq_start_after_trades
        weight_start[i, :] = w

        rr = ret_arr[i, :]
        held = pos > 0.0
        if held.any():
            bad = np.isnan(rr) & held
            if bad.any():
                bad_cols = np.flatnonzero(bad)
                bad_names = [tickers[j] for j in bad_cols[:10]]
                raise ValueError(f"NaN returns for held tickers on {pd.Timestamp(dt).date()}: {bad_names}")

        eq_ret = 0.0
        if eq_start_after_trades > 0.0:
            eq_ret = float(np.dot(w, np.nan_to_num(rr, nan=0.0)))

        pos_end = pos * (1.0 + np.nan_to_num(rr, nan=0.0))
        value_end[i, :] = pos_end
        pos = pos_end

        bench_bucket = bench_bucket * (1.0 + float(bench_ret_arr[i]))
        rf_bucket = rf_bucket * (1.0 + float(rf_ret_arr[i]))

        eq_end = float(pos.sum())

        core.iloc[i, core.columns.get_loc("equity_value_end")] = float(eq_end)
        core.iloc[i, core.columns.get_loc("equity_return")] = float(eq_ret)
        core.iloc[i, core.columns.get_loc("bench_bucket")] = float(bench_bucket)
        core.iloc[i, core.columns.get_loc("rf_bucket")] = float(rf_bucket)
        core.iloc[i, core.columns.get_loc("stocks_plus_cash")] = float(
            eq_end + float(locked_cash) + float(bench_bucket) + float(rf_bucket)
        )

    if not initialized:
        raise ValueError(
            "No valid execution day with a non-empty Top_1 was found; check exec flags and holdings_by_slot_daily Top_1 on those dates"
        )

    exec_log = pd.DataFrame(exec_rows)
    if len(exec_log) == 0:
        raise ValueError("No execution rows were produced; check exec flags and holdings_by_slot_daily content on execution days")

    exec_log["execution_date"] = pd.to_datetime(exec_log["execution_date"])
    exec_log["signal_date_associated"] = pd.to_datetime(exec_log["signal_date_associated"])
    exec_log = exec_log.sort_values("execution_date").set_index("execution_date", drop=False)

    md_exec = md.loc[exec_log.index].copy()
    exec_log = pd.concat([md_exec, exec_log], axis=1)

    value_universe = pd.DataFrame(value_end, index=idx, columns=tickers)
    weight_universe = pd.DataFrame(weight_start, index=idx, columns=tickers)
    ret_universe = r.copy()

    slot_tickers = hs[slot_cols].copy()

    val_slot = np.full((n, len(slot_cols)), np.nan, dtype=float)
    wgt_slot = np.full((n, len(slot_cols)), np.nan, dtype=float)
    ret_slot_arr = np.full((n, len(slot_cols)), np.nan, dtype=float)

    slot_vals = slot_tickers.to_numpy(dtype=object)

    for j in range(len(slot_cols)):
        for i2 in range(n):
            t = slot_vals[i2, j]
            if not (isinstance(t, str) and t):
                continue
            k = colpos.get(t, None)
            if k is None:
                raise ValueError(f"Slot {slot_cols[j]} contains ticker not in ret_ac columns: {t}")
            val_slot[i2, j] = value_end[i2, k]
            wgt_slot[i2, j] = weight_start[i2, k]
            ret_slot_arr[i2, j] = ret_arr[i2, k]

    value_slot = pd.DataFrame(val_slot, index=idx, columns=slot_cols)
    weight_slot = pd.DataFrame(wgt_slot, index=idx, columns=slot_cols)
    ret_slot = pd.DataFrame(ret_slot_arr, index=idx, columns=slot_cols)

    daily = {
        "core": core,
        "ret_universe": ret_universe,
        "value_universe": value_universe,
        "weight_universe": weight_universe,
        "slot_tickers": slot_tickers,
        "ret_slot": ret_slot,
        "value_slot": value_slot,
        "weight_slot": weight_slot,
    }

    return exec_log, daily


def validate_cash_split(cash_reinvest_frac, cash_alloc_bench, cash_alloc_rf, tol=1e-8):
    fracs = {
        "cash_reinvest_frac": float(cash_reinvest_frac),
        "cash_alloc_bench": float(cash_alloc_bench),
        "cash_alloc_rf": float(cash_alloc_rf),
    }
    for name, v in fracs.items():
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {v}")

    total = sum(fracs.values())
    if abs(total - 1.0) > float(tol):
        raise ValueError(
            f"cash_reinvest_frac + cash_alloc_bench + cash_alloc_rf must sum to 1.0, "
            f"got {total:.6g} ({fracs})"
        )
    return fracs


def rolling_std_by_ticker(
    ret_agg,
    lookback,
    ticker_cols=None,
    exclude_cols=None,
    min_periods=None,
    ddof=0,
    shift=1,
    picks_df=None,
    top_prefix="Top_",
):
    df = ret_agg.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[df.index.notna()].sort_index()

    if exclude_cols is None:
        exclude_cols = []

    if ticker_cols is None:
        cand = [c for c in df.columns if c not in set(exclude_cols)]
        tmp = df[cand].apply(pd.to_numeric, errors="coerce")
        ticker_cols = [c for c in tmp.columns if tmp[c].notna().any()]

    x = df[ticker_cols].apply(pd.to_numeric, errors="coerce")

    lb = int(lookback)
    if lb <= 0:
        raise ValueError("lookback must be >= 1")

    if min_periods is None:
        min_periods = lb

    vol = x.rolling(window=lb, min_periods=int(min_periods)).std(ddof=int(ddof))
    if shift:
        vol = vol.shift(int(shift))

    if picks_df is None:
        return vol

    p = picks_df.copy()
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p.loc[p.index.notna()].sort_index()

    top_cols = [c for c in p.columns if str(c).startswith(top_prefix)]
    if len(top_cols) == 0:
        raise ValueError(f"picks_df must contain columns starting with '{top_prefix}'")

    vol_aligned = vol.reindex(p.index)

    out = pd.DataFrame(index=p.index, columns=top_cols, dtype=float)

    for c in top_cols:
        tickers_here = p[c]
        vals = []
        for dt, tkr in tickers_here.items():
            if isinstance(tkr, str) and tkr in vol_aligned.columns:
                vals.append(vol_aligned.at[dt, tkr])
            else:
                vals.append(np.nan)
        out[c] = vals

    return vol, out.dropna()


def build_current_pick_diagnostics(
    signal_res,
    cum_1p,
    ret_ac=None,
    metadata=None,
    inv_vol_weights_exec=None,
    signal_date=None,
    execution_date=None,
    skip_t=None,
    inv_vol_lookback=None,
    inv_vol_skip_t=0,
    inv_vol_min_obs=20,
    inv_vol_ddof=1,
    vol_floor=1e-8,
    top_prefix="Top_",
    include_stage_selected=True,
    include_stage_slot=True,
):
    if not isinstance(signal_res, dict):
        raise TypeError("signal_res must be the dict returned by momentum_filter")

    need = ["picks_df", "tops_map", "cumrets_map", "lookbacks", "top_n"]
    miss = [k for k in need if k not in signal_res]
    if miss:
        raise KeyError(f"signal_res is missing keys: {miss}")

    picks_df = signal_res["picks_df"].copy()
    tops_map = signal_res["tops_map"]
    cumrets_map = signal_res["cumrets_map"]
    lookbacks = [int(x) for x in signal_res["lookbacks"]]
    # signal_res["top_n"] is always momentum_filter's fully-expanded per-stage rank list (even a
    # plain-count stage becomes rank_range(1, n)), so the stage's pick count is len(), not int().
    top_n = [len(x) for x in signal_res["top_n"]]

    if skip_t is None:
        skip_t = int(signal_res.get("skip_t", 0) or 0)
    else:
        skip_t = int(skip_t)

    pk = picks_df.copy()
    if "execution_date" in pk.columns:
        pk["execution_date"] = pd.to_datetime(pk["execution_date"], errors="coerce")
    pk.index = pd.to_datetime(pk.index, errors="coerce")
    pk = pk.loc[pk.index.notna()].sort_index()

    top_cols = [c for c in pk.columns if str(c).startswith(top_prefix)]
    if not top_cols:
        raise ValueError("picks_df has no Top_* columns")

    def _slot_key(c):
        s = str(c)
        n = s.split("_", 1)[1] if "_" in s else ""
        try:
            return int(n)
        except Exception:
            return 10**9

    top_cols = sorted(top_cols, key=_slot_key)

    valid_latest = pk[top_cols[0]].map(lambda x: isinstance(x, str) and len(x) > 0)
    if "is_effective_signal" in pk.columns:
        valid_latest &= pk["is_effective_signal"].fillna(False).astype(bool)
    if "execution_date" in pk.columns:
        valid_latest &= pd.to_datetime(pk["execution_date"], errors="coerce").notna()

    latest_valid_signal_dt = pk.index[valid_latest][-1] if valid_latest.any() else pd.NaT

    if signal_date is not None:
        signal_dt = pd.Timestamp(signal_date)
        if signal_dt not in pk.index:
            raise KeyError(f"signal_date {signal_dt} not found in picks_df index")
    elif execution_date is not None:
        execution_dt = pd.Timestamp(execution_date)
        if "execution_date" not in pk.columns:
            raise KeyError("execution_date passed but picks_df has no execution_date column")
        mask = pd.to_datetime(pk["execution_date"]).eq(execution_dt)
        if not mask.any():
            raise KeyError(f"execution_date {execution_dt} not found in picks_df")
        signal_dt = pk.index[mask][-1]
    else:
        valid = pk[top_cols[0]].map(lambda x: isinstance(x, str) and len(x) > 0)
        if "is_effective_signal" in pk.columns:
            valid &= pk["is_effective_signal"].fillna(False).astype(bool)
        if "execution_date" in pk.columns:
            valid &= pd.to_datetime(pk["execution_date"], errors="coerce").notna()
        if not valid.any():
            raise ValueError("No valid current pick row found in picks_df")
        signal_dt = pk.index[valid][-1]

    is_last_pick = pd.notna(latest_valid_signal_dt) and (pd.Timestamp(signal_dt) == pd.Timestamp(latest_valid_signal_dt))

    row = pk.loc[signal_dt]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]

    execution_dt = pd.to_datetime(row.get("execution_date", pd.NaT), errors="coerce")
    selected = [x for x in row[top_cols].tolist() if isinstance(x, str) and x]
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("Selected row has no picked tickers")

    cum_df = cum_1p.copy()
    if "period_end" not in cum_df.columns:
        raise KeyError("cum_1p must contain 'period_end'")
    cum_df["period_end"] = pd.to_datetime(cum_df["period_end"], errors="coerce")
    cum_df = cum_df.loc[cum_df["period_end"].notna()].sort_values("period_end")
    signal_dates = pd.DatetimeIndex(cum_df["period_end"].values)
    if signal_dt not in signal_dates:
        raise KeyError(f"signal_date {signal_dt} not found in cum_1p period_end")

    onep = cum_df[selected].copy()
    onep.index = signal_dates

    stage_candidates = []
    prev = None
    for s, (lb, N) in enumerate(zip(lookbacks, top_n), start=1):
        stage_name = f"stage{s}"
        if stage_name not in tops_map:
            raise KeyError(f"tops_map missing key '{stage_name}'")

        stage_df = tops_map[stage_name].copy()
        stage_df.index = pd.to_datetime(stage_df.index, errors="coerce")
        stage_df = stage_df.loc[stage_df.index.notna()].sort_index()
        if signal_dt not in stage_df.index:
            stage_candidates.append([] if prev is None else prev)
            prev = []
            continue

        if prev is None:
            cand = list(cumrets_map[int(lb)].columns)
            meta_like = {
                "period_start", "period_end", "signal_creation_start_date", "signal_creation_end_date",
                "signal_date", "execution_date", "actual_signal_date", "actual_execution_date",
                "is_effective_signal"
            }
            cand = [c for c in cand if c in onep.columns and c not in meta_like]
        else:
            cand = prev.copy()

        stage_candidates.append(cand)
        stage_top_cols = [c for c in stage_df.columns if str(c).startswith(top_prefix)]
        stage_top_cols = sorted(stage_top_cols, key=_slot_key)
        prev = [x for x in stage_df.loc[signal_dt, stage_top_cols].tolist() if isinstance(x, str) and x]
        prev = list(dict.fromkeys(prev))

    long_rows = []

    inv_w_row = None
    inv_w_row_has_weights = False

    if inv_vol_weights_exec is not None and pd.notna(execution_dt):
        wex = inv_vol_weights_exec.copy()
        wex.index = pd.to_datetime(wex.index, errors="coerce")
        wex = wex.loc[wex.index.notna()].sort_index()

        if execution_dt in wex.index:
            inv_w_row = wex.loc[execution_dt]
            if isinstance(inv_w_row, pd.DataFrame):
                inv_w_row = inv_w_row.iloc[-1]

            chk = pd.to_numeric(
                pd.Series({tk: inv_w_row.get(tk, np.nan) for tk in selected}),
                errors="coerce"
            )
            inv_w_row_has_weights = chk.notna().any()

    vol_stats = {}
    fallback_inv_w = {}

    if ret_ac is not None and metadata is not None and inv_vol_lookback is not None:
        r = ret_ac.copy()
        r.index = pd.to_datetime(r.index, errors="coerce")
        r = r.loc[r.index.notna()].sort_index()

        md = metadata.copy()
        md.index = pd.to_datetime(md.index, errors="coerce")
        md = md.loc[md.index.notna()].sort_index()

        common = r.index.intersection(md.index)
        r = r.loc[common]
        md = md.loc[common]

        if "is_signal_day" not in md.columns:
            raise KeyError("metadata must contain is_signal_day for inverse-vol diagnostics")

        sig_dates_md = pd.DatetimeIndex(
            md.index[md["is_signal_day"].fillna(False).astype(bool).values]
        ).sort_values().unique()
        sig_arr = sig_dates_md.values

        sig_dt_for_vol = pd.NaT

        if is_last_pick:
            sig_dt_for_vol = pd.Timestamp(signal_dt)
        else:
            sig_col = "actual_signal_date" if "actual_signal_date" in md.columns else "signal_date"
            if pd.notna(execution_dt) and execution_dt in md.index:
                sig_dt_for_vol = pd.to_datetime(md.loc[execution_dt, sig_col], errors="coerce")

        if pd.notna(sig_dt_for_vol):
            t = np.searchsorted(sig_arr, sig_dt_for_vol.to_datetime64(), side="left")
            if t >= len(sig_arr) or pd.Timestamp(sig_arr[t]) != pd.Timestamp(sig_dt_for_vol):
                t = np.searchsorted(sig_arr, sig_dt_for_vol.to_datetime64(), side="right") - 1

            L = int(inv_vol_lookback)
            stv = int(inv_vol_skip_t)
            end_pos = t - stv
            start_boundary_pos = t - L

            if end_pos >= 0 and start_boundary_pos >= 0 and end_pos > start_boundary_pos:
                start_boundary = pd.Timestamp(sig_arr[start_boundary_pos])
                end_dt = pd.Timestamp(sig_arr[end_pos])
                win = r.loc[(r.index > start_boundary) & (r.index <= end_dt), selected].copy()

                used_std_map = {}

                for tk in selected:
                    s = pd.to_numeric(win[tk], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if len(s) >= int(inv_vol_min_obs):
                        raw_std = float(s.std(ddof=int(inv_vol_ddof)))
                        used_std = max(raw_std, float(vol_floor)) if np.isfinite(raw_std) and raw_std > 0 else np.nan
                    else:
                        raw_std = np.nan
                        used_std = np.nan

                    vol_stats[tk] = {
                        "invvol_window_start": start_boundary,
                        "invvol_window_end": end_dt,
                        "invvol_obs": int(len(s)),
                        "invvol_std_raw": raw_std,
                        "invvol_std_used": used_std,
                    }
                    used_std_map[tk] = used_std

                if not inv_w_row_has_weights:
                    finite_used = [v for v in used_std_map.values() if np.isfinite(v) and v > 0]

                    if len(finite_used) == 0:
                        fallback_inv_w = {tk: 1.0 / float(len(selected)) for tk in selected}
                    else:
                        fill_v = float(np.median(finite_used))
                        vol_vec = np.array(
                            [
                                used_std_map[tk] if (np.isfinite(used_std_map[tk]) and used_std_map[tk] > 0) else fill_v
                                for tk in selected
                            ],
                            dtype=float,
                        )
                        vol_vec = np.maximum(vol_vec, float(vol_floor))
                        inv = 1.0 / vol_vec
                        inv_sum = float(np.sum(inv))

                        if (not np.isfinite(inv_sum)) or inv_sum <= 0:
                            fallback_inv_w = {tk: 1.0 / float(len(selected)) for tk in selected}
                        else:
                            w_raw = inv / inv_sum
                            fallback_inv_w = {tk: float(w_raw[i]) for i, tk in enumerate(selected)}

    tpos = signal_dates.get_loc(signal_dt)

    for slot, tk in enumerate(selected, start=1):
        rec = {
            "ticker": tk,
            "slot_final": int(slot),
            "signal_date": signal_dt,
            "execution_date": execution_dt,
        }

        for s, (lb, N) in enumerate(zip(lookbacks, top_n), start=1):
            L = int(lb)
            stage_name = f"stage{s}"
            cand = stage_candidates[s - 1]

            cr_stage = cumrets_map[L].copy()
            cr_stage.index = pd.to_datetime(cr_stage.index, errors="coerce")
            cr_stage = cr_stage.loc[cr_stage.index.notna()].sort_index()

            stage_rank = np.nan
            stage_cumret = np.nan
            stage_selected = False
            stage_slot = np.nan

            if signal_dt in cr_stage.index and tk in cr_stage.columns and cand:
                svals = pd.to_numeric(cr_stage.loc[signal_dt, cand], errors="coerce").dropna().sort_values(ascending=False)
                if tk in svals.index:
                    stage_rank = float(pd.Series(np.arange(1, len(svals) + 1), index=svals.index)[tk])
                    stage_cumret = float(pd.to_numeric(cr_stage.loc[signal_dt, tk], errors="coerce"))

            stage_df = tops_map[stage_name].copy()
            stage_df.index = pd.to_datetime(stage_df.index, errors="coerce")
            stage_df = stage_df.loc[stage_df.index.notna()].sort_index()
            stage_top_cols = [c for c in stage_df.columns if str(c).startswith(top_prefix)]
            stage_top_cols = sorted(stage_top_cols, key=_slot_key)
            if signal_dt in stage_df.index:
                stage_list = [x for x in stage_df.loc[signal_dt, stage_top_cols].tolist() if isinstance(x, str) and x]
                stage_list = list(dict.fromkeys(stage_list))
                if tk in stage_list:
                    stage_selected = True
                    stage_slot = float(stage_list.index(tk) + 1)

            window_vals = pd.Series(dtype=float)
            end_i = tpos - int(skip_t)
            start_i = end_i - L + 1
            if start_i >= 0 and end_i >= start_i and tk in onep.columns:
                window_vals = pd.to_numeric(onep.iloc[start_i:end_i + 1][tk], errors="coerce").dropna()

            rec[f"stage{s}_lb"] = L
            rec[f"stage{s}_topn"] = int(N)
            rec[f"stage{s}_rank"] = stage_rank
            rec[f"stage{s}_cumret"] = stage_cumret
            rec[f"stage{s}_window_mean"] = float(window_vals.mean()) if len(window_vals) else np.nan
            rec[f"stage{s}_window_median"] = float(window_vals.median()) if len(window_vals) else np.nan
            rec[f"stage{s}_window_p10"] = float(window_vals.quantile(0.10)) if len(window_vals) else np.nan
            rec[f"stage{s}_window_p90"] = float(window_vals.quantile(0.90)) if len(window_vals) else np.nan
            rec[f"stage{s}_window_obs"] = int(len(window_vals))
            if include_stage_selected:
                rec[f"stage{s}_selected"] = bool(stage_selected)
            if include_stage_slot:
                rec[f"stage{s}_slot"] = stage_slot

        if inv_w_row_has_weights:
            rec["inv_vol_weight"] = float(pd.to_numeric(inv_w_row.get(tk, np.nan), errors="coerce"))
        elif tk in fallback_inv_w:
            rec["inv_vol_weight"] = float(fallback_inv_w.get(tk, np.nan))
        else:
            rec["inv_vol_weight"] = np.nan

        if tk in vol_stats:
            rec.update(vol_stats[tk])
        else:
            rec["invvol_window_start"] = pd.NaT
            rec["invvol_window_end"] = pd.NaT
            rec["invvol_obs"] = np.nan
            rec["invvol_std_raw"] = np.nan
            rec["invvol_std_used"] = np.nan

        long_rows.append(rec)

    long_df = pd.DataFrame(long_rows)
    wide_df = long_df.set_index("ticker").T
    meta = {
        "signal_date": signal_dt,
        "execution_date": execution_dt,
        "selected_tickers": selected,
        "lookbacks": lookbacks,
        "top_n": top_n,
        "skip_t": int(skip_t),
    }
    return wide_df, long_df, meta


def create_metadata(usable_sample, freq, signal_weekday=None, exec_weekday=None, hold_period=1):
    idx = pd.DatetimeIndex(pd.to_datetime(usable_sample, errors="coerce")).dropna().sort_values().unique()
    if len(idx) == 0:
        raise ValueError("usable_sample is empty")

    f = str(freq).strip().lower()
    if f in ("weekly", "week", "w"):
        f = "weekly"
    elif f in ("monthly", "month", "m"):
        f = "monthly"
    else:
        raise ValueError("freq must be 'weekly' or 'monthly'")

    hp = int(hold_period)
    if hp < 1:
        raise ValueError("hold_period must be >= 1")

    out = pd.DataFrame(index=idx)
    out.index.name = "Date"

    out["weekday_num"] = out.index.weekday + 1
    out["weekday"] = out.index.day_name().str[:3]

    months = out.index.to_period("M")
    s = pd.Series(out.index.values, index=months)
    first_m = s.groupby(level=0).min().reindex(months).to_numpy()
    last_m = s.groupby(level=0).max().reindex(months).to_numpy()
    out["cal_is_month_first_obs"] = out.index.values == first_m
    out["cal_is_month_last_obs"] = out.index.values == last_m

    bool_cols = [
        "is_signal_day",
        "is_execution_day",
        "is_actual_execution_day",
        "is_actual_signal_day",
    ]
    for c in bool_cols:
        out[c] = False

    dt_cols = [
        "signal_date",
        "execution_date",
        "execution_date_associated_signal_date",
        "signal_date_associated_with_execution_date",
        "actual_signal_date",
        "actual_execution_date",
        "actual_execution_date_associated_signal_date",
        "actual_signal_date_associated_with_execution_date",
    ]
    for c in dt_cols:
        out[c] = pd.NaT

    def _weekday_1based(x):
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            xi = int(x)
            if 1 <= xi <= 7:
                return xi
            raise ValueError(f"weekday int must be in 1..7. Got {xi}")
        s2 = str(x).strip().lower()
        mp = {
            "mon": 1, "monday": 1,
            "tue": 2, "tues": 2, "tuesday": 2,
            "wed": 3, "wednesday": 3,
            "thu": 4, "thur": 4, "thurs": 4, "thursday": 4,
            "fri": 5, "friday": 5,
            "sat": 6, "saturday": 6,
            "sun": 7, "sunday": 7,
        }
        if s2 not in mp:
            raise ValueError(f"Unrecognized weekday: {x}")
        return mp[s2]

    idx_arr = idx.values
    max_dt = pd.Timestamp(idx.max())

    def _last_available_on_or_before(ts):
        if pd.isna(ts):
            return pd.NaT
        tsv = pd.Timestamp(ts).to_datetime64()
        pos = np.searchsorted(idx_arr, tsv, side="right") - 1
        if pos < 0:
            return pd.NaT
        return pd.Timestamp(idx_arr[pos])

    def _first_available_on_or_after(ts):
        if pd.isna(ts):
            return pd.NaT
        tsv = pd.Timestamp(ts).to_datetime64()
        pos = np.searchsorted(idx_arr, tsv, side="left")
        if pos >= len(idx_arr):
            return pd.NaT
        return pd.Timestamp(idx_arr[pos])

    if f == "weekly":
        sw = _weekday_1based(signal_weekday)
        ew = _weekday_1based(exec_weekday)
        if sw is None or ew is None:
            raise ValueError("weekly freq requires signal_weekday and exec_weekday")

        weeks = pd.period_range(idx.min().to_period("W-SUN"), idx.max().to_period("W-SUN"), freq="W-SUN")
        nominal_signal_dates = pd.DatetimeIndex([wk.start_time.normalize() + pd.Timedelta(days=sw - 1) for wk in weeks])
        nominal_execution_dates = pd.DatetimeIndex([wk.start_time.normalize() + pd.Timedelta(days=ew - 1) for wk in weeks])

        cfg_signal = str(signal_weekday)
        cfg_exec = str(exec_weekday)

    else:
        months_full = pd.period_range(idx.min().to_period("M"), idx.max().to_period("M"), freq="M")
        nominal_signal_dates = pd.DatetimeIndex([m.to_timestamp(how="end").normalize() for m in months_full])
        nominal_execution_dates = pd.DatetimeIndex([m.to_timestamp(how="start").normalize() for m in months_full])

        cfg_signal = "Month_End"
        cfg_exec = "Month_Start"

    nominal_signal_dates = nominal_signal_dates[nominal_signal_dates <= max_dt]

    if len(nominal_signal_dates) == 0:
        raise ValueError("No signal days produced")

    sig_nom_arr = nominal_signal_dates.values
    exe_nom_arr = nominal_execution_dates.values

    records = []
    for ns in nominal_signal_dates:
        pos_next_exec = np.searchsorted(exe_nom_arr, ns.to_datetime64(), side="right")
        nominal_exec = pd.Timestamp(exe_nom_arr[pos_next_exec]) if pos_next_exec < len(exe_nom_arr) else pd.NaT

        actual_sig = _last_available_on_or_before(ns)
        actual_exec = _first_available_on_or_after(nominal_exec) if pd.notna(nominal_exec) else pd.NaT

        records.append(
            {
                "nominal_signal_date": pd.Timestamp(ns),
                "actual_signal_date": pd.Timestamp(actual_sig) if pd.notna(actual_sig) else pd.NaT,
                "nominal_execution_date": pd.Timestamp(nominal_exec) if pd.notna(nominal_exec) else pd.NaT,
                "actual_execution_date": pd.Timestamp(actual_exec) if pd.notna(actual_exec) else pd.NaT,
            }
        )

    sched = pd.DataFrame(records)
    sched = sched.loc[sched["actual_signal_date"].notna()].copy()
    sched = sched.sort_values(["nominal_signal_date", "actual_signal_date"]).reset_index(drop=True)

    keep_mask = sched["actual_signal_date"].shift().isna() | sched["actual_signal_date"].gt(sched["actual_signal_date"].shift())
    sched = sched.loc[keep_mask].reset_index(drop=True)

    if len(sched) == 0:
        raise ValueError("No valid signal schedule produced")

    sched["execution_date_effective"] = sched["actual_execution_date"].combine_first(sched["nominal_execution_date"])

    sched_exec = sched.loc[sched["actual_execution_date"].notna()].copy()
    if len(sched_exec) > 0:
        sched_exec = sched_exec.drop_duplicates(subset=["actual_execution_date"], keep="last").reset_index(drop=True)

    selected = sched.iloc[::hp].copy().reset_index(drop=True)
    selected_exec = selected.loc[selected["actual_execution_date"].notna()].copy()
    if len(selected_exec) > 0:
        selected_exec = selected_exec.drop_duplicates(subset=["actual_execution_date"], keep="last").reset_index(drop=True)

    for _, r in sched.iterrows():
        sdt = pd.Timestamp(r["actual_signal_date"])
        edt_eff = r["execution_date_effective"]
        edt_eff = pd.Timestamp(edt_eff) if pd.notna(edt_eff) else pd.NaT

        out.loc[sdt, "is_signal_day"] = True
        out.loc[sdt, "signal_date"] = sdt
        out.loc[sdt, "execution_date"] = edt_eff
        out.loc[sdt, "execution_date_associated_signal_date"] = edt_eff
        out.loc[sdt, "signal_date_associated_with_execution_date"] = sdt

    for _, r in sched_exec.iterrows():
        sdt = pd.Timestamp(r["actual_signal_date"])
        edt = pd.Timestamp(r["actual_execution_date"])

        out.loc[edt, "is_execution_day"] = True
        out.loc[edt, "signal_date"] = sdt
        out.loc[edt, "execution_date"] = edt
        out.loc[edt, "execution_date_associated_signal_date"] = edt
        out.loc[edt, "signal_date_associated_with_execution_date"] = sdt

    for _, r in selected.iterrows():
        sdt = pd.Timestamp(r["actual_signal_date"])
        aedt = r["actual_execution_date"]
        aedt = pd.Timestamp(aedt) if pd.notna(aedt) else pd.NaT

        out.loc[sdt, "is_actual_signal_day"] = True
        out.loc[sdt, "actual_signal_date"] = sdt
        out.loc[sdt, "actual_execution_date"] = aedt
        out.loc[sdt, "actual_execution_date_associated_signal_date"] = aedt
        out.loc[sdt, "actual_signal_date_associated_with_execution_date"] = sdt

    for _, r in selected_exec.iterrows():
        sdt = pd.Timestamp(r["actual_signal_date"])
        aedt = pd.Timestamp(r["actual_execution_date"])

        out.loc[aedt, "is_actual_execution_day"] = True
        out.loc[aedt, "actual_signal_date"] = sdt
        out.loc[aedt, "actual_execution_date"] = aedt
        out.loc[aedt, "actual_execution_date_associated_signal_date"] = aedt
        out.loc[aedt, "actual_signal_date_associated_with_execution_date"] = sdt

    for c in dt_cols:
        out[c] = pd.to_datetime(out[c], errors="coerce")

    out.insert(0, "cfg_hold_period", hp)
    out.insert(0, "cfg_exec_day", cfg_exec)
    out.insert(0, "cfg_signal_day", cfg_signal)
    out.insert(0, "cfg_freq", f)

    return out


def aggregate_to_freq(df, metadata):
    d = df.copy()
    m = metadata.copy()

    d.index = pd.to_datetime(d.index, errors="coerce")
    m.index = pd.to_datetime(m.index, errors="coerce")

    d = d.loc[d.index.notna()].sort_index()
    m = m.loc[m.index.notna()].sort_index()

    common = d.index.intersection(m.index)
    if len(common) == 0:
        raise ValueError("No overlapping dates between df and metadata")

    d = d.loc[common]
    m = m.loc[common]

    sig_flag = "is_actual_signal_day" if "is_actual_signal_day" in m.columns else "is_signal_day"
    exec_flag = "is_actual_execution_day" if "is_actual_execution_day" in m.columns else "is_execution_day"

    if sig_flag not in m.columns or exec_flag not in m.columns:
        raise KeyError("metadata must contain signal and execution flags")

    meta_cols = set(m.columns.tolist())
    ret_cols = [c for c in d.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(d[c])]
    if len(ret_cols) == 0:
        raise ValueError("No numeric columns to aggregate")

    signal_dates = pd.DatetimeIndex(m.index[m[sig_flag].fillna(False).astype(bool)]).sort_values().unique()
    execution_dates = pd.DatetimeIndex(m.index[m[exec_flag].fillna(False).astype(bool)]).sort_values().unique()

    if len(signal_dates) == 0:
        raise ValueError(f"No signal days in metadata using {sig_flag}")
    if len(execution_dates) == 0:
        raise ValueError(f"No execution days in metadata using {exec_flag}")

    exec_arr = execution_dates.values

    rows = []
    starts = []
    ends = []
    meta_rows = []
    out_index = []

    for s in signal_dates:
        pos_prev_exec = np.searchsorted(exec_arr, s.to_datetime64(), side="left") - 1
        if pos_prev_exec < 0:
            continue

        e_prev = pd.Timestamp(exec_arr[pos_prev_exec])
        if not (e_prev < s):
            continue

        win = d.loc[(d.index >= e_prev) & (d.index <= s), ret_cols].astype(float)
        if len(win) == 0:
            continue

        cnt = win.notna().sum(axis=0)
        x = (1.0 + win.fillna(0.0)).prod(axis=0) - 1.0
        x = x.where(cnt > 0, np.nan)

        rows.append(x)
        starts.append(e_prev)
        ends.append(s)
        out_index.append(s)
        meta_rows.append(m.loc[s])

    if len(rows) == 0:
        raise ValueError("No valid execution->signal windows found")

    out = pd.DataFrame(rows, index=pd.DatetimeIndex(out_index, name="Date"))
    out.insert(0, "period_start", pd.to_datetime(starts))
    out.insert(1, "period_end", pd.to_datetime(ends))

    meta_end = pd.DataFrame(meta_rows, index=out.index)
    out = pd.concat([meta_end, out], axis=1)

    keep = list(meta_end.columns) + ["period_start", "period_end"]
    rest = [c for c in out.columns if c not in keep]
    out = out[keep + rest]

    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    return out


def compute_inv_vol_weights_on_exec(
    ret_ac,
    metadata,
    holdings_by_slot_daily,
    top_k,
    lookback=12,
    frequency=None,
    skip_t=0,
    min_obs=20,
    vol_floor=1e-8,
    ddof=1,
    exec_flag=None,
    use_signal_col=None,
    top_prefix="Top_",
    include_metadata=True,
    metadata_cols=None,
    check_sum_to_one=True,
    sum_tol=1e-8,
    picks_df=None,
):
    r = ret_ac.copy()
    r.index = pd.to_datetime(r.index, errors="coerce")
    r = r.loc[~r.index.isna()].sort_index()

    md = metadata.copy()
    md.index = pd.to_datetime(md.index, errors="coerce")
    md = md.loc[~md.index.isna()].sort_index()

    hs = holdings_by_slot_daily.copy()
    hs.index = pd.to_datetime(hs.index, errors="coerce")
    hs = hs.loc[~hs.index.isna()].sort_index()

    K = int(top_k)
    if K <= 0:
        raise ValueError("top_k must be >= 1")

    L = int(lookback)
    if L <= 0:
        raise ValueError("lookback must be >= 1")

    st = int(skip_t)
    if st < 0:
        raise ValueError("skip_t must be >= 0")
    if st >= L:
        raise ValueError(f"skip_t ({st}) must be < lookback ({L})")

    mo = int(min_obs)
    if mo < 2:
        raise ValueError("min_obs must be >= 2")

    vf = float(vol_floor)
    if vf <= 0:
        raise ValueError("vol_floor must be > 0")

    if exec_flag is None:
        exec_flag = "is_actual_execution_day" if "is_actual_execution_day" in md.columns else "is_execution_day"
    if exec_flag not in md.columns:
        raise KeyError(f"metadata must contain '{exec_flag}'")

    if "is_signal_day" not in md.columns:
        raise KeyError("metadata must contain 'is_signal_day'")

    if frequency is None:
        if "cfg_freq" in md.columns:
            vals = pd.Series(md["cfg_freq"]).dropna().astype(str).str.lower().unique().tolist()
            vals = [v for v in vals if v]
            if len(vals) == 1:
                frequency = vals[0]
    if frequency is not None:
        f = str(frequency).strip().lower()
        if f in ("weekly", "week", "w"):
            f = "weekly"
        elif f in ("monthly", "month", "m"):
            f = "monthly"
        else:
            raise ValueError("frequency must be 'weekly' or 'monthly'")
        if "cfg_freq" in md.columns:
            vals = pd.Series(md["cfg_freq"]).dropna().astype(str).str.lower().unique().tolist()
            vals = [v for v in vals if v]
            if len(vals) == 1 and vals[0] != f:
                raise ValueError(f"frequency='{f}' does not match metadata cfg_freq='{vals[0]}'")

    if use_signal_col is None:
        use_signal_col = "actual_signal_date" if "actual_signal_date" in md.columns else "signal_date"
    if use_signal_col not in md.columns and "signal_date" not in md.columns:
        raise KeyError("metadata must contain 'signal_date' (and optionally 'actual_signal_date')")

    slot_cols = [c for c in hs.columns if str(c).startswith(top_prefix)]
    if len(slot_cols) == 0:
        raise ValueError(f"holdings_by_slot_daily must contain columns starting with '{top_prefix}'")

    def _slot_key(c):
        s = str(c)
        n = s.split("_", 1)[1] if "_" in s else ""
        try:
            return int(n)
        except Exception:
            return 10**9

    slot_cols = sorted(slot_cols, key=_slot_key)[:K]
    if len(slot_cols) < K:
        raise ValueError(f"holdings_by_slot_daily has only {len(slot_cols)} {top_prefix}* columns but top_k={K}")

    tickers = list(r.columns)
    if len(tickers) == 0:
        raise ValueError("ret_ac has no ticker columns")

    idx = md.index.intersection(hs.index).intersection(r.index)
    idx = pd.DatetimeIndex(idx).sort_values()
    if len(idx) == 0:
        raise ValueError("No overlapping dates among ret_ac, metadata, holdings_by_slot_daily")

    r2 = r.loc[idx]
    md2 = md.loc[idx]
    hs2 = hs.loc[idx]

    sig_dates = pd.DatetimeIndex(md2.index[md2["is_signal_day"].fillna(False).astype(bool).values]).sort_values().unique()
    if len(sig_dates) == 0:
        raise ValueError("No signal days found in metadata (is_signal_day)")

    sig_arr = sig_dates.values
    exec_dates = pd.DatetimeIndex(md2.index[md2[exec_flag].fillna(False).astype(bool).values]).sort_values().unique()

    w_daily_vals = np.full((len(idx), len(tickers)), np.nan, dtype=float)
    weights_daily_vals = np.full((len(idx), len(slot_cols)), np.nan, dtype=float)

    idx_pos = {dt: i for i, dt in enumerate(idx)}
    pos_map = {t: j for j, t in enumerate(tickers)}

    w_exec_rows = []
    w_exec_index = []
    diag_rows = []

    def _append_weight_row(exd, row_slots, sig_dt, write_daily=True):
        held = [x for x in row_slots if isinstance(x, str) and x]

        if len(held) > 0 and len(set(held)) != len(held):
            vc = pd.Series(held).value_counts()
            dups = vc[vc > 1].index.tolist()
            raise ValueError(f"Duplicate tickers across slots on {pd.Timestamp(exd).date()}: {dups}")

        if len(held) == 0:
            w_vec = np.zeros(len(tickers), dtype=float)
            w_exec_rows.append(w_vec)
            w_exec_index.append(exd)
            if write_daily and exd in idx_pos:
                w_daily_vals[idx_pos[exd], :] = w_vec
            diag_rows.append(
                dict(
                    execution_date=exd,
                    signal_date=pd.NaT,
                    window_start=pd.NaT,
                    window_end=pd.NaT,
                    n_tickers=0,
                    note="no_holdings",
                )
            )
            return

        sig_dt = pd.to_datetime(sig_dt, errors="coerce")
        if pd.isna(sig_dt):
            w_vec = np.full(len(tickers), np.nan, dtype=float)
            w_exec_rows.append(w_vec)
            w_exec_index.append(exd)
            if write_daily and exd in idx_pos:
                w_daily_vals[idx_pos[exd], :] = w_vec
            diag_rows.append(
                dict(
                    execution_date=exd,
                    signal_date=pd.NaT,
                    window_start=pd.NaT,
                    window_end=pd.NaT,
                    n_tickers=len(held),
                    note="missing_signal_date",
                )
            )
            return

        t = np.searchsorted(sig_arr, sig_dt.to_datetime64(), side="left")
        if t >= len(sig_arr) or pd.Timestamp(sig_arr[t]) != pd.Timestamp(sig_dt):
            t = np.searchsorted(sig_arr, sig_dt.to_datetime64(), side="right") - 1

        end_pos = t - st
        start_boundary_pos = t - L
        if end_pos < 0 or start_boundary_pos < 0 or end_pos <= start_boundary_pos:
            w_vec = np.full(len(tickers), np.nan, dtype=float)
            w_exec_rows.append(w_vec)
            w_exec_index.append(exd)
            if write_daily and exd in idx_pos:
                w_daily_vals[idx_pos[exd], :] = w_vec
            diag_rows.append(
                dict(
                    execution_date=exd,
                    signal_date=sig_dt,
                    window_start=pd.NaT,
                    window_end=pd.NaT,
                    n_tickers=len(held),
                    note="insufficient_history",
                )
            )
            return

        start_boundary = pd.Timestamp(sig_arr[start_boundary_pos])
        end_dt = pd.Timestamp(sig_arr[end_pos])

        for tkr in held:
            if tkr not in pos_map:
                raise ValueError(f"Ticker '{tkr}' not found in ret_ac columns")

        win = r2.loc[(r2.index > start_boundary) & (r2.index <= end_dt), held].copy()
        if win.empty:
            w_vec = np.full(len(tickers), np.nan, dtype=float)
            w_exec_rows.append(w_vec)
            w_exec_index.append(exd)
            if write_daily and exd in idx_pos:
                w_daily_vals[idx_pos[exd], :] = w_vec
            diag_rows.append(
                dict(
                    execution_date=exd,
                    signal_date=sig_dt,
                    window_start=start_boundary,
                    window_end=end_dt,
                    n_tickers=len(held),
                    note="empty_window",
                )
            )
            return

        vols = {}
        for tkr in held:
            s = pd.to_numeric(win[tkr], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) >= mo:
                v = float(s.std(ddof=int(ddof)))
                vols[tkr] = max(v, vf) if (np.isfinite(v) and v > 0) else np.nan
            else:
                vols[tkr] = np.nan

        finite_vols = [v for v in vols.values() if np.isfinite(v) and v > 0]
        if len(finite_vols) == 0:
            w_sub = {tkr: 1.0 / float(len(held)) for tkr in held}
        else:
            fill_v = float(np.median(finite_vols))
            vol_vec = np.array(
                [vols[tkr] if (np.isfinite(vols[tkr]) and vols[tkr] > 0) else fill_v for tkr in held],
                dtype=float,
            )
            vol_vec = np.maximum(vol_vec, vf)
            inv = 1.0 / vol_vec
            s_inv = float(np.sum(inv))
            if (not np.isfinite(s_inv)) or s_inv <= 0:
                w_sub = {tkr: 1.0 / float(len(held)) for tkr in held}
            else:
                w_raw = inv / s_inv
                w_sub = {tkr: float(w_raw[k]) for k, tkr in enumerate(held)}

        w_vec = np.zeros(len(tickers), dtype=float)
        for tkr, w in w_sub.items():
            w_vec[pos_map[tkr]] = float(w)

        if check_sum_to_one:
            ssum = float(np.sum([w_sub[tkr] for tkr in held]))
            if (not np.isfinite(ssum)) or abs(ssum - 1.0) > float(sum_tol):
                raise ValueError(f"Ticker weights do not sum to 1 on {pd.Timestamp(exd).date()}: sum={ssum}")

        slot_w = np.full(len(slot_cols), np.nan, dtype=float)
        for j, tk in enumerate(row_slots[:len(slot_cols)]):
            if isinstance(tk, str) and tk:
                slot_w[j] = float(w_sub.get(tk, np.nan))

        if check_sum_to_one:
            ssw = float(np.nansum(slot_w))
            if (not np.isfinite(ssw)) or abs(ssw - 1.0) > float(sum_tol):
                raise ValueError(f"Slot weights do not sum to 1 on {pd.Timestamp(exd).date()}: sum={ssw}")

        w_exec_rows.append(w_vec)
        w_exec_index.append(exd)

        if write_daily and exd in idx_pos:
            i = idx_pos[exd]
            w_daily_vals[i, :] = w_vec
            weights_daily_vals[i, :] = slot_w

        diag_rows.append(
            dict(
                execution_date=exd,
                signal_date=sig_dt,
                window_start=start_boundary,
                window_end=end_dt,
                n_tickers=len(held),
                note="",
            )
        )

    for exd in exec_dates:
        if exd not in idx_pos:
            continue

        row_slots = hs2.loc[exd, slot_cols].tolist()
        sig_dt = md2.loc[exd, use_signal_col] if use_signal_col in md2.columns else pd.NaT
        if pd.isna(sig_dt) and "signal_date" in md2.columns:
            sig_dt = md2.loc[exd, "signal_date"]

        _append_weight_row(exd=exd, row_slots=row_slots, sig_dt=sig_dt, write_daily=True)

    if picks_df is not None:
        pk = picks_df.copy()
        pk.index = pd.to_datetime(pk.index, errors="coerce")
        pk = pk.loc[~pk.index.isna()].sort_index()

        if "execution_date" not in pk.columns:
            raise KeyError("picks_df must contain 'execution_date'")

        pk_slot_cols = [c for c in pk.columns if str(c).startswith(top_prefix)]
        pk_slot_cols = sorted(pk_slot_cols, key=_slot_key)[:K]
        if len(pk_slot_cols) == 0:
            raise ValueError(f"picks_df must contain columns starting with '{top_prefix}'")

        valid = pd.to_datetime(pk["execution_date"], errors="coerce").notna()
        valid &= pk[pk_slot_cols[0]].map(lambda x: isinstance(x, str) and len(x) > 0)

        if "is_effective_signal" in pk.columns:
            valid &= pk["is_effective_signal"].fillna(False).astype(bool)

        pkv = pk.loc[valid].copy()
        if len(pkv) > 0:
            last_sig_dt = pkv.index[-1]
            last_row = pkv.iloc[-1]

            pending_exd = pd.to_datetime(last_row.get("execution_date", pd.NaT), errors="coerce")
            if pd.notna(pending_exd) and pending_exd not in pd.DatetimeIndex(w_exec_index):
                row_slots = [last_row.get(c, np.nan) for c in pk_slot_cols]
                _append_weight_row(
                    exd=pending_exd,
                    row_slots=row_slots,
                    sig_dt=last_sig_dt,
                    write_daily=(pending_exd in idx_pos),
                )

    weights_exec = pd.DataFrame(
        w_exec_rows,
        index=pd.DatetimeIndex(w_exec_index, name="execution_date"),
        columns=tickers,
        dtype=float,
    ).sort_index()

    weights_daily = pd.DataFrame(w_daily_vals, index=idx, columns=tickers, dtype=float)
    weights_df = pd.DataFrame(weights_daily_vals, index=idx, columns=slot_cols, dtype=float)

    diag_df = pd.DataFrame(diag_rows)
    if len(diag_df) > 0:
        diag_df["execution_date"] = pd.to_datetime(diag_df["execution_date"])
        diag_df = diag_df.sort_values("execution_date").set_index("execution_date", drop=True)

    if include_metadata:
        meta_daily = md2.copy() if metadata_cols is None else md2[list(metadata_cols)].copy()
        weights_daily = pd.concat([meta_daily, weights_daily], axis=1)

        meta_source = md.copy() if metadata_cols is None else md[list(metadata_cols)].copy()
        meta_exec = meta_source.reindex(weights_exec.index)
        weights_exec = pd.concat([meta_exec, weights_exec], axis=1)

    return weights_exec, weights_daily, weights_df, diag_df
