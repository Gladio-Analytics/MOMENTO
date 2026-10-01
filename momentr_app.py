import pandas as pd
import streamlit as st

import momentum_backend as backend
import momentum_fetch as fetch
import momentum_viz as viz

st.set_page_config(page_title="MOMENTR", layout="wide")

UNIVERSE_TO_INDICES = {
    ("US", "mega"): ["sp500", "djia", "ndx"],
    ("US", "sp500"): ["sp500"],
    ("US", "djia"): ["djia"],
    ("US", "ndx"): ["ndx"],
    ("CAN", "tsx60"): ["tsx60"],
}

PRESETS = {
    "A_let_it_run": dict(winner_handling="let_ride", loser_handling="leave_alone", position_sizing="equal", cash_reinvest_frac=1.0),
    "B_fixed_stake": dict(winner_handling="trim_to_target", loser_handling="top_up_to_target", position_sizing="equal", cash_reinvest_frac=0.0),
    "C_reinvest_all": dict(winner_handling="trim_to_target", loser_handling="top_up_to_target", position_sizing="inverse_vol", cash_reinvest_frac=1.0),
    "D_harvest_above_stake": dict(winner_handling="trim_to_target", loser_handling="top_up_to_target", position_sizing="inverse_vol", cash_reinvest_frac=0.0),
    "E_equal_reinvested": dict(winner_handling="trim_to_target", loser_handling="top_up_to_target", position_sizing="equal", cash_reinvest_frac=1.0),
    "F_partial_reinvest_50": dict(winner_handling="trim_to_target", loser_handling="top_up_to_target", position_sizing="equal", cash_reinvest_frac=0.5),
}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

DECISION_TREE_URL = "https://claude.ai/artifact/AWTgYb3d1AD9Z4kJGja4sG"

PROGRESS_STAGES = [
    "Fetching universe prices",
    "Fetching reference prices",
    "Preparing samples & calendar",
    "Building momentum signal",
    "Computing portfolio weights",
    "Running ledger simulation",
    "Building performance diagnostics",
    "Building current-selection diagnostics",
]

DEFAULTS = {
    "winner_handling": "let_ride",
    "loser_handling": "leave_alone",
    "position_sizing": "equal",
    "cash_reinvest_frac": 1.0,
    "entry_funding": "capped_entry",
    "entry_ramp_frac": 1.0,
    "bench_share": 0.9,
    "preset_choice": "Custom",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_data(show_spinner=False)
def load_universe_prices(market, universe):
    indices = UNIVERSE_TO_INDICES[(market, universe)]
    return fetch.load_price_data(indices=indices)


@st.cache_data(show_spinner=False)
def load_reference_prices(passive_ticker, active_ticker, rf_ticker):
    tickers = [passive_ticker, active_ticker, rf_ticker]
    reference_close, success, failed = backend.fetch_prices(tickers, keep_survivors=True, ffill_prices=True)
    if failed:
        fail_info = reference_close.attrs.get("fail_info", {})
        details = "; ".join(f"{t}: {fail_info.get(t, 'unknown reason')}" for t in failed)
        raise RuntimeError(f"Could not fetch reference ticker(s) from Yahoo Finance - {details}")
    return backend.prep_reference(reference_close, tickers)


def apply_preset():
    choice = st.session_state["preset_choice"]
    if choice == "Custom":
        return
    preset = PRESETS[choice]
    st.session_state["winner_handling"] = preset["winner_handling"]
    st.session_state["loser_handling"] = preset["loser_handling"]
    st.session_state["position_sizing"] = preset["position_sizing"]
    st.session_state["cash_reinvest_frac"] = preset["cash_reinvest_frac"]


def parse_int_list(raw, n_expected, field_name):
    if raw is None or not str(raw).strip():
        return None, f"{field_name} is required"
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    try:
        vals = [int(p) for p in parts]
    except ValueError:
        return None, f"{field_name} must be a comma-separated list of whole numbers"
    if len(vals) != n_expected:
        return None, f"{field_name} needs exactly {n_expected} value(s) (got {len(vals)})"
    if any(v <= 0 for v in vals):
        return None, f"{field_name} values must all be positive"
    return vals, None


st.title("MOMENTR")
st.caption("Cross-sectional momentum backtest dashboard")

with st.sidebar:
    st.header("Universe")
    market = st.selectbox("Market", ["US", "CAN"])
    universe_opts = ["sp500", "djia", "ndx", "mega"] if market == "US" else ["tsx60"]
    universe = st.selectbox("Universe", universe_opts)

    st.header("Calendar")
    start_date = st.date_input("Backtest start date", value=pd.Timestamp("2010-01-01"))
    freq = st.selectbox("Rebalance frequency", ["weekly", "monthly"])
    if freq == "weekly":
        signal_weekday = st.selectbox("Signal weekday", WEEKDAYS, index=4)
        exec_weekday = st.selectbox("Execution weekday", WEEKDAYS, index=0)
    else:
        signal_weekday = None
        exec_weekday = None
    hold_period = st.number_input("Hold period", min_value=1, value=1, step=1)

    st.header("Reference tickers")
    passive_bench_ticker = st.text_input("Passive benchmark ticker", value="SPY")
    active_bench_ticker = st.text_input("Active benchmark ticker", value="SPY")
    riskfree_ticker = st.text_input("Risk-free ticker", value="BIL")

    st.header("Capital & costs")
    notional = st.number_input("Starting notional", min_value=0.0, value=1.0, step=0.1)
    slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, value=10.0, step=1.0)

with st.expander("Decision tree reference"):
    st.markdown(f"Full interactive version: [{DECISION_TREE_URL}]({DECISION_TREE_URL})")
    st.markdown(
        """
1. **Universe** - market -> which tickers are eligible at all (sidebar)
2. **Calendar** - rebalance frequency, signal/exec weekday, hold period (sidebar)
3. **Momentum signal** - lookback windows + top-N per cascade stage, and `skip_t`
4. **Weighting scheme** - equal vs. inverse-volatility
5. **Winner handling** - trim an overweight re-selected holding, or let it ride
6. **Loser handling** - top up an underweight re-selected holding, or leave it alone (independent of 5)
7. **Entry funding** - fund a brand-new entrant fully, or cap/ramp it in
8. **Profit handling** - reinvest fraction of profit above the ratcheting notional watermark; the rest splits between the benchmark and risk-free sleeves
9. **Trading costs** - slippage (sidebar)
10. **Reference tickers** - for reporting only, doesn't affect returns (sidebar)
11. **Starting capital** - the notional (sidebar)
        """
    )

st.subheader("1. Momentum signal")
n_stages_label = st.radio("Momentum stages", ["Single", "Dual", "Triple"], index=2, horizontal=True)
n_stages = {"Single": 1, "Dual": 2, "Triple": 3}[n_stages_label]

col1, col2 = st.columns(2)
with col1:
    lookbacks_str = st.text_input(f"Lookback windows, widest first - {n_stages} value(s), comma-separated", value="")
with col2:
    top_n_str = st.text_input(f"Top-N counts per stage - {n_stages} value(s), comma-separated", value="")

skip_t = st.number_input(
    "skip_t - periods to skip before measuring the lookback return (0 includes the most recent period)",
    min_value=0, value=0, step=1,
)

st.subheader("2. Weighting & rebalance behavior")
st.selectbox("Preset (optional shortcut)", ["Custom"] + list(PRESETS.keys()), key="preset_choice", on_change=apply_preset)

pcol1, pcol2 = st.columns(2)
with pcol1:
    position_sizing = st.radio(
        "Weighting scheme", ["equal", "inverse_vol"], key="position_sizing", horizontal=True,
        format_func=lambda x: "Equal weight" if x == "equal" else "Inverse-volatility weight",
    )
    if position_sizing == "inverse_vol":
        inv_vol_lookback = st.number_input(
            "Inverse-vol lookback (rebalance periods - independent of the momentum cascade's own "
            "lookbacks; too short and every ticker can fall short of the 20-observation minimum, "
            "silently collapsing to equal weight)",
            min_value=4, value=12, step=1,
        )
    else:
        inv_vol_lookback = None
    winner_handling = st.radio(
        "Winner handling", ["let_ride", "trim_to_target"], key="winner_handling", horizontal=True,
        format_func=lambda x: "Let it ride" if x == "let_ride" else "Trim to target",
    )
    loser_handling = st.radio(
        "Loser handling", ["leave_alone", "top_up_to_target"], key="loser_handling", horizontal=True,
        format_func=lambda x: "Leave alone" if x == "leave_alone" else "Top up to target",
    )
with pcol2:
    entry_funding = st.radio(
        "Entry funding", ["full_target", "capped_entry"], key="entry_funding", horizontal=True,
        format_func=lambda x: "Full target" if x == "full_target" else "Capped entry",
    )
    if entry_funding == "capped_entry":
        entry_ramp_frac = st.slider("Entry ramp fraction", 0.01, 1.0, key="entry_ramp_frac", step=0.01)
    else:
        entry_ramp_frac = 1.0

    cash_reinvest_frac = st.slider(
        "Reinvest fraction of profit above the notional watermark", 0.0, 1.0, key="cash_reinvest_frac", step=0.05,
    )
    remainder = 1.0 - cash_reinvest_frac
    if remainder > 1e-9:
        bench_share = st.slider(
            "Of the non-reinvested share: fraction to benchmark sleeve (rest to risk-free)",
            0.0, 1.0, key="bench_share", step=0.05,
        )
    else:
        bench_share = st.session_state["bench_share"]
    cash_alloc_bench = remainder * bench_share
    cash_alloc_rf = remainder * (1.0 - bench_share)

lookbacks, lb_err = parse_int_list(lookbacks_str, n_stages, "Lookback windows")
top_n_counts, tn_err = parse_int_list(top_n_str, n_stages, "Top-N counts")

errors = []
if lb_err:
    errors.append(lb_err)
if tn_err:
    errors.append(tn_err)
if top_n_counts and len(top_n_counts) > 1:
    for i in range(1, len(top_n_counts)):
        if top_n_counts[i] > top_n_counts[i - 1]:
            errors.append("Top-N counts must be non-increasing stage to stage (each stage narrows the field)")
            break
if not passive_bench_ticker.strip() or not active_bench_ticker.strip() or not riskfree_ticker.strip():
    errors.append("All three reference tickers are required")

for e in errors:
    st.warning(e)

run_clicked = st.button("Run backtest", type="primary", disabled=len(errors) > 0)

if run_clicked:
    progress_bar = st.progress(0.0, text=PROGRESS_STAGES[0])
    try:
        adj_close = load_universe_prices(market, universe)
        progress_bar.progress(1 / len(PROGRESS_STAGES), text=PROGRESS_STAGES[1])

        reference_close = load_reference_prices(passive_bench_ticker, active_bench_ticker, riskfree_ticker)
        progress_bar.progress(2 / len(PROGRESS_STAGES), text=PROGRESS_STAGES[2])

        available_sample = backend.get_available_sample(adj_close, reference_close)
        usable_sample = backend.get_usable_sample(available_sample, str(start_date))

        adj_close_t = backend.trim_adj_close_prices(adj_close, usable_sample)
        reference_close_t = backend.trim_adj_close_prices(reference_close, usable_sample)
        adj_close_filled = backend.check_gaps(adj_close_t, fill_na_gaps=True)["df"]
        reference_close_filled = backend.check_gaps(reference_close_t, fill_na_gaps=True)["df"]
        ret_ac = backend.to_returns(adj_close_filled)
        ret_ref = backend.to_returns(reference_close_filled)

        metadata = backend.create_metadata(usable_sample, freq, signal_weekday, exec_weekday, hold_period)
        ret_ac_with_meta_agg = backend.aggregate_to_freq(backend.add_metadata(ret_ac, metadata), metadata)
        ret_ref_with_meta = backend.add_metadata(ret_ref, metadata)
        last_data_date = adj_close.index.max()
        progress_bar.progress(3 / len(PROGRESS_STAGES), text=PROGRESS_STAGES[3])

        mf = backend.momentum_filter(ret_ac_with_meta_agg, lookbacks=lookbacks, top_n=top_n_counts, skip_t=int(skip_t))
        holdings_universe_daily, holdings_by_slot_daily = backend.populate_portfolio(
            metadata=metadata, picks_df=mf["picks_df"], universe=ret_ac.columns,
        )
        t0 = backend.get_t0(metadata, holdings_by_slot_daily)
        progress_bar.progress(4 / len(PROGRESS_STAGES), text=PROGRESS_STAGES[4])

        weights_exec = None
        if position_sizing == "inverse_vol":
            weights_exec, _, _, _ = backend.compute_inv_vol_weights_on_exec(
                ret_ac=ret_ac, metadata=metadata, holdings_by_slot_daily=holdings_by_slot_daily,
                top_k=top_n_counts[-1], lookback=inv_vol_lookback,
            )
        progress_bar.progress(5 / len(PROGRESS_STAGES), text=PROGRESS_STAGES[5])

        bench_col = f"Active Benchmark ({active_bench_ticker.strip().upper()})"
        rf_col = f"Risk-Free Rate ({riskfree_ticker.strip().upper()})"

        exec_log, daily = backend.simulate_portfolio_ledger(
            ret_ac=ret_ac, metadata=metadata, holdings_by_slot_daily=holdings_by_slot_daily,
            notional=notional, top_k=top_n_counts[-1], slippage_bps=slippage_bps,
            winner_handling=winner_handling, loser_handling=loser_handling,
            entry_funding=entry_funding, position_sizing=position_sizing,
            inv_vol_weights_exec=weights_exec,
            cash_reinvest_frac=cash_reinvest_frac, cash_alloc_bench=cash_alloc_bench, cash_alloc_rf=cash_alloc_rf,
            entry_ramp_frac=entry_ramp_frac,
            bench_ret=ret_ref[bench_col], rf_ret=ret_ref[rf_col],
            check_cash=True, cash_tol=1e-6,
        )
        progress_bar.progress(6 / len(PROGRESS_STAGES), text=PROGRESS_STAGES[6])

        core_series = backend.extract_core_portfolio_series(daily)
        strategy_df = backend.create_strategy_value(core_series=core_series, ret_ref_with_meta=ret_ref_with_meta)
        strategy_df_cut = backend.cut_strategy_df(strategy_df, t0)
        strategy_df_cut_cum = backend.add_reference_cumret_columns(strategy_df_cut, ret_ref)

        notional_df = backend.extract_notional_series(daily)
        notional_df_cut = backend.cut_strategy_df(notional_df, t0)

        buys_df, sells_df = backend.get_buy_and_sell_df(
            metadata=metadata, holdings_by_slot_daily=holdings_by_slot_daily,
            top_k=top_n_counts[-1], slippage_bps=slippage_bps,
        )
        # cut to t0: before the strategy's first real execution, equity_value_end is a flat 0.0
        # (nothing bought yet during the lookback warm-up), which otherwise shows up as a bogus
        # "-notional" equity-sleeve drawdown and distorts the heatmap's return stats.
        equity_sleeve_ret = backend.cut_strategy_df(
            backend.strategy_return_from_weight_and_return_df(daily["weight_slot"], daily["ret_slot"]), t0,
        )
        ret_slot_with_meta = backend.merge_ret_slot_with_metadata_from_t0(daily["ret_slot"], metadata, t0)
        simple_ret_df = backend.cut_strategy_df(
            backend.create_simple_returns_df(
                ret_slot=daily["ret_slot"], core_series=core_series, ref_returns=ret_ref_with_meta,
            ),
            t0,
        )
        opf_weekly = backend.ret_to_overperformance_df(simple_ret_df, notional=notional, agg="weekly")

        panel = backend.build_heatmap_panel(
            simple_ret_df=simple_ret_df, equity_sleeve_ret=equity_sleeve_ret, strategy_df_cut=strategy_df_cut,
        )
        agg_month = backend.aggregate_heatmap_panel(panel["returns_panel"], by="month")
        agg_weekday = backend.aggregate_heatmap_panel(panel["returns_panel"], by="weekday")
        hist = backend.cumulative_top_histories(ret_slot_with_meta)
        progress_bar.progress(7 / len(PROGRESS_STAGES), text=PROGRESS_STAGES[7])

        wide_df, long_df, current_pick_meta = backend.build_current_pick_diagnostics(
            signal_res=mf, cum_1p=ret_ac_with_meta_agg, ret_ac=ret_ac, metadata=metadata,
            inv_vol_weights_exec=weights_exec, inv_vol_lookback=inv_vol_lookback, inv_vol_skip_t=0,
        )
        progress_bar.progress(1.0, text="Done")

        st.session_state["results"] = dict(
            strategy_df_cut_cum=strategy_df_cut_cum,
            strategy_df_cut=strategy_df_cut,
            notional_df_cut=notional_df_cut,
            mf=mf, metadata=metadata, ret_ac_with_meta_agg=ret_ac_with_meta_agg,
            lookbacks=lookbacks, top_n_counts=top_n_counts, skip_t=int(skip_t),
            daily=daily, buys_df=buys_df, sells_df=sells_df, strategy_df=strategy_df,
            opf_weekly=opf_weekly, agg_month=agg_month, agg_weekday=agg_weekday, hist=hist,
            wide_df=wide_df, long_df=long_df,
            final_value=float(core_series["stocks_plus_cash"].iloc[-1]),
            t0=t0, last_data_date=last_data_date,
            n_trades=int(exec_log["n_buys"].sum() + exec_log["n_sells"].sum()),
        )
        st.success("Backtest complete and cash-conservation verified.")
    except Exception as e:
        st.session_state["results"] = None
        st.error(f"Backtest failed: {e}")

results = st.session_state.get("results")
if results is not None:
    st.subheader("Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final portfolio value", f"{results['final_value']:.4f}")
    m2.metric("Backtest start (t0)", pd.Timestamp(results["t0"]).date().isoformat())
    m3.metric("Total trades", results["n_trades"])
    m4.metric("Last data date", pd.Timestamp(results["last_data_date"]).date().isoformat())

    tab_overview, tab_current, tab_hist = st.tabs(["Strategy Overview", "Current Selection", "Historical Performance"])

    with tab_overview:
        st.plotly_chart(viz.plot_strategy_overview_plotly(results["strategy_df_cut_cum"]), use_container_width=True)
        st.plotly_chart(viz.plot_notional_watermark(results["notional_df_cut"]), use_container_width=True)

    with tab_current:
        st.plotly_chart(
            viz.plot_top_stocks_from_ret(
                returns=results["ret_ac_with_meta_agg"], picks_df=results["mf"]["picks_df"], metadata=results["metadata"],
                lookbacks=results["lookbacks"], topNs=results["top_n_counts"], skip_t=results["skip_t"],
                returns_date_col="period_end", picks_date_col="signal_date", execution_col="execution_date",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            viz.plot_topn_empirical_and_stats_panel(results["hist"], ncols=5, min_obs=5),
            use_container_width=True,
        )
        st.markdown("Tickers currently selected, by slot, with the lookback/top-N that produced each stage's cut.")
        st.dataframe(results["wide_df"], use_container_width=True)
        with st.expander("Long-format detail"):
            st.dataframe(results["long_df"], use_container_width=True)

    with tab_hist:
        st.plotly_chart(viz.plot_return_distribution_scatter_dropdown(results["strategy_df_cut"]), use_container_width=True)
        st.plotly_chart(
            viz.plot_topn_clean_drilldown(
                picks_df=results["mf"]["picks_df"], metadata=results["metadata"],
                date_col="signal_date", execution_col="execution_date",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            viz.plot_topn_value_and_excess_over_time(
                slot_value_df=results["daily"]["value_slot"], ret_slot_df=results["daily"]["ret_slot"],
                passive_ret=results["strategy_df_cut"], value_indexed=True,
            ),
            use_container_width=True,
        )
        fig_skim, _ = viz.plot_turnover_slippage_skim_info(
            buys_df=results["buys_df"], sells_df=results["sells_df"], core_series=results["strategy_df"],
        )
        st.plotly_chart(fig_skim, use_container_width=True)
        st.plotly_chart(viz.plot_strategy_performance_relative_to_benchmark(results["opf_weekly"]), use_container_width=True)
        st.plotly_chart(
            viz.plot_heatmap_panel(results["agg_month"], value="mean", gap_rows=1, title="Monthly Mean Returns"),
            use_container_width=True,
        )
        st.plotly_chart(
            viz.plot_heatmap_panel(results["agg_weekday"], value="mean", gap_rows=1, title="Weekday Mean Returns"),
            use_container_width=True,
        )
