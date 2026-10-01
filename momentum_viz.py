import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_strategy_overview_plotly(strategy_df_cut_cum, suffix="_cumret"):
    df = strategy_df_cut_cum.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    def _pick_one(prefix, end=None):
        cols = [c for c in df.columns if str(c).startswith(prefix)]
        if end is not None:
            cols = [c for c in cols if str(c).endswith(end)]
        if len(cols) == 0:
            return None
        return cols[0]

    passive_cum_col = _pick_one("Passive Benchmark (", suffix)
    if passive_cum_col is None:
        raise KeyError(f"Missing passive benchmark cumulative column ending with '{suffix}'")

    rf_cum_col = _pick_one("Risk-Free Rate (", suffix)
    if rf_cum_col is None:
        rf_raw_col = _pick_one("Risk-Free Rate (")
        if rf_raw_col is None:
            raise KeyError("Missing Risk-Free Rate column")
        rf_cum = (1.0 + pd.to_numeric(df[rf_raw_col], errors="coerce")).cumprod() - 1.0
    else:
        rf_cum = pd.to_numeric(df[rf_cum_col], errors="coerce")

    strategy_val_col = "value_stocks_plus_cash" if "value_stocks_plus_cash" in df.columns else "stocks_plus_cash"
    needed = [strategy_val_col, "equity_value_end", "value_active_benchmark", "value_riskfree_rate"]
    missing = [c for c in needed if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns: {missing}")

    s_strategy_val = pd.to_numeric(df[strategy_val_col], errors="coerce")
    base = s_strategy_val.replace(0, np.nan).dropna()
    if len(base) == 0:
        raise ValueError("No positive/nonzero strategy base value found")
    base = float(base.iloc[0])

    strategy = s_strategy_val / base
    passive = 1.0 + pd.to_numeric(df[passive_cum_col], errors="coerce")
    rf_rate = 1.0 + rf_cum

    stocks = pd.to_numeric(df["equity_value_end"], errors="coerce") / base
    active_sleeve = pd.to_numeric(df["value_active_benchmark"], errors="coerce") / base
    rf_sleeve = pd.to_numeric(df["value_riskfree_rate"], errors="coerce") / base

    def _dd(x):
        x = pd.to_numeric(x, errors="coerce")
        return x / x.cummax() - 1.0

    dd_strategy = _dd(strategy)
    dd_passive = _dd(passive)

    strategy_color = "#1f77b4"
    sleeve_color = "#6baed6"
    passive_color = "#d62728"
    rf_color = "#000000"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.68, 0.32],
        subplot_titles=("Cumulative Returns", "Drawdowns"),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=strategy, mode="lines", name="Strategy",
            line=dict(color=strategy_color, width=2.4)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=stocks, mode="lines", name="Stocks Sleeve",
            line=dict(color=sleeve_color, width=1.8, dash="dash")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=active_sleeve, mode="lines", name="Active Sleeve",
            line=dict(color=sleeve_color, width=1.8, dash="dot")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=rf_sleeve, mode="lines", name="Risk-Free Sleeve",
            line=dict(color=sleeve_color, width=1.8, dash="dashdot")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=passive, mode="lines", name="Passive Benchmark",
            line=dict(color=passive_color, width=2.0)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=rf_rate, mode="lines", name="Risk-Free Rate",
            line=dict(color=rf_color, width=1.9, dash="solid")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=dd_strategy, mode="lines", name="Strategy Drawdown",
            line=dict(color=strategy_color, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.15)",
            showlegend=False
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=dd_passive, mode="lines", name="Passive Benchmark Drawdown",
            line=dict(color=passive_color, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(214,39,40,0.10)",
            showlegend=False
        ),
        row=2, col=1
    )

    fig.update_yaxes(title_text="Cumulative Returns", row=1, col=1)
    fig.update_yaxes(title_text="Drawdowns", tickformat=".0%", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=780,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.0
        ),
        margin=dict(l=60, r=30, t=80, b=40),
    )

    return fig


def plot_notional_watermark(notional_df, title="Notional Watermark"):
    df = notional_df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()

    need = ["current_notional", "stocks_plus_cash", "total_skimmed"]
    missing = [c for c in need if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns: {missing}")

    value_color = "#1f77b4"
    notional_color = "#d62728"
    gap_fill = "rgba(31,119,180,0.12)"
    skim_color = "#2ca02c"
    skim_fill = "rgba(44,160,44,0.15)"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.68, 0.32],
        subplot_titles=("Portfolio Value vs. Notional Watermark", "Cumulative Skim (Benchmark + Risk-Free)"),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["stocks_plus_cash"], mode="lines", name="Portfolio Value",
            line=dict(color=value_color, width=2.2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["current_notional"], mode="lines", name="Notional Watermark",
            line=dict(color=notional_color, width=1.8, shape="hv"),
            fill="tonexty", fillcolor=gap_fill,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["total_skimmed"], mode="lines", name="Total Skimmed",
            line=dict(color=skim_color, width=1.8),
            fill="tozeroy", fillcolor=skim_fill,
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Skimmed", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=680,
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=90, b=40),
    )

    return fig


def plot_yoy_rolling_strategy_return(
    strategy_df,
    value_col=None,
    ret_col=None,
    window=252,
    title="Year-over-Year Rolling Cumulative Return (Strategy)",
):
    df = strategy_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if ret_col is None:
        if value_col is None:
            if "value_stocks_plus_cash" in df.columns:
                value_col = "value_stocks_plus_cash"
            elif "stocks_plus_cash" in df.columns:
                value_col = "stocks_plus_cash"
            else:
                raise KeyError("No strategy value column found. Pass value_col or ret_col.")
        v = pd.to_numeric(df[value_col], errors="coerce")
        if v.dropna().empty:
            raise ValueError(f"Column '{value_col}' has no numeric data")
        r = v.pct_change()
    else:
        if ret_col not in df.columns:
            raise KeyError(f"ret_col '{ret_col}' not in strategy_df")
        r = pd.to_numeric(df[ret_col], errors="coerce")

    roll = (1.0 + r).rolling(int(window), min_periods=int(window)).apply(np.prod, raw=True) - 1.0
    neg = roll.where(roll < 0.0, 0.0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=neg,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.18)",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=roll,
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            showlegend=False,
            name="Strategy",
        )
    )

    fig.add_hline(y=0.0, line_dash="dash", line_color="black", line_width=1)

    fig.update_layout(
        template="plotly",
        title=title,
        xaxis_title="Date",
        yaxis_title="Rolling compounded return",
        height=500,
        margin=dict(t=60, b=50, l=65, r=30),
    )

    return fig


def plot_return_distribution_scatter_dropdown(
    strategy_df,
    passive_col=None,
    strategy_ret_col=None,
    title="Return distributions + scatter (Strategy vs Passive)",
):
    df = strategy_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if passive_col is None:
        cands = [c for c in df.columns if str(c).startswith("Passive Benchmark (") and not str(c).endswith("_cumret")]
        if len(cands) == 0:
            raise KeyError("No raw passive benchmark return column found")
        passive_col = cands[0]

    if str(passive_col).endswith("_cumret"):
        raise ValueError("passive_col must be raw returns, not cumulative returns")

    x = pd.to_numeric(df[passive_col], errors="coerce")

    if strategy_ret_col is not None:
        if strategy_ret_col not in df.columns:
            raise KeyError(f"strategy_ret_col '{strategy_ret_col}' not found")
        y_strategy = pd.to_numeric(df[strategy_ret_col], errors="coerce")
    else:
        if "value_stocks_plus_cash" in df.columns:
            y_strategy = pd.to_numeric(df["value_stocks_plus_cash"], errors="coerce").pct_change()
        elif "stocks_plus_cash" in df.columns:
            y_strategy = pd.to_numeric(df["stocks_plus_cash"], errors="coerce").pct_change()
        else:
            raise KeyError("Need strategy_ret_col or value_stocks_plus_cash/stocks_plus_cash")

    series_map = {"Strategy": y_strategy}

    if "equity_return" in df.columns:
        y = pd.to_numeric(df["equity_return"], errors="coerce")
        if "equity_value_end" in df.columns:
            y = y.where(pd.to_numeric(df["equity_value_end"], errors="coerce") > 0)
        series_map["Stocks Sleeve"] = y

    if "ret_active_benchmark" in df.columns:
        y = pd.to_numeric(df["ret_active_benchmark"], errors="coerce")
        if "into_active_benchmark" in df.columns:
            y = y.where(pd.to_numeric(df["into_active_benchmark"], errors="coerce") > 0)
        series_map["Active Sleeve"] = y

    if "ret_riskfree_rate" in df.columns:
        y = pd.to_numeric(df["ret_riskfree_rate"], errors="coerce")
        if "into_riskfree_rate" in df.columns:
            y = y.where(pd.to_numeric(df["into_riskfree_rate"], errors="coerce") > 0)
        series_map["Risk-Free Sleeve"] = y

    mode_names = [k for k, v in series_map.items() if v.notna().sum() >= 5]
    if len(mode_names) == 0:
        raise ValueError("No valid return series available")

    def _kde(vals, grid):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        n = vals.size
        if n < 2:
            return np.full(grid.shape, np.nan)
        sd = float(np.std(vals, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1e-6
        bw = 1.06 * sd * (n ** (-1.0 / 5.0))
        bw = max(float(bw), 1e-6)
        z = (grid[:, None] - vals[None, :]) / bw
        return np.exp(-0.5 * z * z).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))

    pool = [x.values] + [series_map[k].values for k in mode_names]
    pool = np.concatenate(pool)
    pool = pool[np.isfinite(pool)]
    lo = np.quantile(pool, 0.005)
    hi = np.quantile(pool, 0.995)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(pool)), float(np.max(pool))
        if lo == hi:
            lo, hi = lo - 0.01, hi + 0.01
    pad = 0.08 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 400)

    x_kde = _kde(x.values, grid)

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.20,
        subplot_titles=("Kernel density (returns)", "Scatter (returns)"),
    )

    fig.add_trace(
        go.Scatter(
            x=grid, y=x_kde, mode="lines",
            name=str(passive_col),
            line=dict(color="#E24A33", width=2),
            fill="tozeroy", fillcolor="rgba(226,74,51,0.20)"
        ),
        row=1, col=1
    )

    trace_map = {}
    for i, name in enumerate(mode_names):
        y = pd.to_numeric(series_map[name], errors="coerce")

        y_kde = _kde(y.values, grid)
        fig.add_trace(
            go.Scatter(
                x=grid, y=y_kde, mode="lines",
                name=name,
                line=dict(color="#4C6EF5", width=2),
                fill="tozeroy", fillcolor="rgba(76,110,245,0.20)",
                visible=(i == 0)
            ),
            row=1, col=1
        )
        kde_idx = len(fig.data) - 1

        xy = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        fig.add_trace(
            go.Scatter(
                x=xy["x"], y=xy["y"], mode="markers",
                marker=dict(color="rgba(170,170,170,0.55)", size=5),
                showlegend=False,
                visible=(i == 0)
            ),
            row=1, col=2
        )
        sc_idx = len(fig.data) - 1

        if len(xy) >= 3 and xy["x"].nunique() > 1:
            beta, alpha = np.polyfit(xy["x"].values, xy["y"].values, 1)
            corr = float(np.corrcoef(xy["x"].values, xy["y"].values)[0, 1])
            xline = np.linspace(float(xy["x"].min()), float(xy["x"].max()), 120)
            yline = alpha + beta * xline
            text = f"corr = {corr:.3f}<br>y = {alpha:.6f} + {beta:.3f}x"
            tx = float(np.quantile(xy["x"], 0.03))
            ty = float(np.quantile(xy["y"], 0.95))
        else:
            xline = np.array([])
            yline = np.array([])
            text = "corr = NA<br>y = NA"
            tx = 0.0
            ty = 0.0

        fig.add_trace(
            go.Scatter(
                x=xline, y=yline, mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                showlegend=False, visible=(i == 0)
            ),
            row=1, col=2
        )
        fit_idx = len(fig.data) - 1

        fig.add_trace(
            go.Scatter(
                x=[tx], y=[ty], mode="text",
                text=[text], textposition="top left",
                textfont=dict(size=18, color="#2a3f5f"),
                showlegend=False, hoverinfo="skip",
                visible=(i == 0)
            ),
            row=1, col=2
        )
        txt_idx = len(fig.data) - 1

        trace_map[name] = [kde_idx, sc_idx, fit_idx, txt_idx]

    total = len(fig.data)
    buttons = []
    for name in mode_names:
        vis = [False] * total
        vis[0] = True
        for j in trace_map[name]:
            vis[j] = True
        buttons.append(
            dict(
                label=name,
                method="update",
                args=[
                    {"visible": vis},
                    {
                        "title": f"Return distributions + scatter ({name} vs Passive)",
                        "yaxis2": {"title": {"text": name}}
                    },
                ],
            )
        )

    fig.add_hline(y=0, row=1, col=2, line_dash="dash", line_color="black", line_width=1)
    fig.add_vline(x=0, row=1, col=2, line_dash="dash", line_color="black", line_width=1)

    fig.update_xaxes(title_text="Return", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text=str(passive_col), row=1, col=2)
    fig.update_yaxes(title_text=mode_names[0], row=1, col=2)

    fig.update_layout(
        title=f"Return distributions + scatter ({mode_names[0]} vs Passive)",
        template="plotly",
        height=620,
        margin=dict(t=80, b=50, l=65, r=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
        updatemenus=[dict(type="dropdown", buttons=buttons, x=1.02, y=1.07)],
    )

    return fig


def plot_topn_slot_value_over_time(
    slot_value_df,
    date_col=None,
    slot_prefix="Top_",
    indexed=True,
    title="Top-N slot value over time",
    top_colors=None,
):
    df = slot_value_df.copy()

    if date_col is not None and date_col in df.columns:
        df.index = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()

    slot_cols = [c for c in df.columns if str(c).startswith(slot_prefix)]

    def _k(c):
        s = str(c).split("_", 1)
        if len(s) == 2 and s[1].isdigit():
            return int(s[1])
        return 10**9

    slot_cols = sorted(slot_cols, key=_k)
    if len(slot_cols) == 0:
        raise ValueError(f"No slot columns found with prefix '{slot_prefix}'")

    val = df[slot_cols].apply(pd.to_numeric, errors="coerce")
    slot_cols = [c for c in slot_cols if val[c].notna().any()]
    if len(slot_cols) == 0:
        raise ValueError("All slot columns are empty/non-numeric")

    val = val[slot_cols]

    if top_colors is None:
        top_colors = ["#7C3AED", "#F97316", "#06B6D4", "#0F766E", "#64748B",
                      "#A855F7", "#22C55E", "#EAB308", "#EC4899", "#14B8A6", "#F59E0B"]

    K = len(slot_cols)
    fig = make_subplots(
        rows=K,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"Rank {i}" for i in range(1, K + 1)],
    )

    for i, c in enumerate(slot_cols, start=1):
        s = val[c].copy()

        if indexed:
            idx0 = s.first_valid_index()
            if idx0 is not None:
                b = s.loc[idx0]
                if pd.notna(b) and b != 0:
                    s = s / b
                else:
                    s = s * np.nan

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                line=dict(width=2, color=top_colors[(i - 1) % len(top_colors)]),
                name=f"Rank {i}",
                showlegend=False,
            ),
            row=i,
            col=1,
        )

        if indexed:
            fig.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="black",
                line_width=1,
                row=i,
                col=1,
            )

        fig.update_yaxes(
            title_text="Indexed (start=1)" if indexed else "Dollar value",
            row=i,
            col=1,
        )

    fig.update_xaxes(title_text="Date", row=K, col=1)

    fig.update_layout(
        title=title,
        template="plotly",
        height=max(260 * K, 500),
        margin=dict(t=70, b=50, l=70, r=30),
    )

    return fig


def plot_topn_slot_excess_over_passive(
    ret_slot_df,
    passive_ret,
    date_col=None,
    slot_prefix="Top_",
    title="Top-N slot excess return over passive benchmark",
    top_colors=None,
):
    rs = ret_slot_df.copy()
    if date_col is not None and date_col in rs.columns:
        rs.index = pd.to_datetime(rs[date_col], errors="coerce")
    else:
        rs.index = pd.to_datetime(rs.index, errors="coerce")
    rs = rs.loc[~rs.index.isna()].sort_index()

    slot_cols = [c for c in rs.columns if str(c).startswith(slot_prefix)]

    def _k(c):
        s = str(c).split("_", 1)
        return int(s[1]) if len(s) == 2 and s[1].isdigit() else 10**9

    slot_cols = sorted(slot_cols, key=_k)
    if len(slot_cols) == 0:
        raise ValueError(f"No slot columns found with prefix '{slot_prefix}'")

    slots = rs[slot_cols].apply(pd.to_numeric, errors="coerce")

    if isinstance(passive_ret, pd.Series):
        p = pd.to_numeric(passive_ret.copy(), errors="coerce")
        p.index = pd.to_datetime(p.index, errors="coerce")
    else:
        pr = passive_ret.copy()
        if date_col is not None and date_col in pr.columns:
            pr.index = pd.to_datetime(pr[date_col], errors="coerce")
        else:
            pr.index = pd.to_datetime(pr.index, errors="coerce")
        pr = pr.loc[~pr.index.isna()].sort_index()

        candidates = [c for c in pr.columns if ("Passive" in str(c)) and (not str(c).endswith("_cumret"))]
        if len(candidates) == 0:
            raise KeyError("No raw passive column found containing 'Passive' and not ending with '_cumret'")
        p = pd.to_numeric(pr[candidates[0]], errors="coerce")

    idx = slots.index.intersection(p.index)
    if len(idx) == 0:
        raise ValueError("No overlapping dates between ret_slot_df and passive_ret")

    excess = slots.loc[idx].sub(p.loc[idx], axis=0)

    if top_colors is None:
        top_colors = ["#7C3AED", "#F97316", "#06B6D4", "#0F766E", "#64748B",
                      "#A855F7", "#22C55E", "#EAB308", "#EC4899", "#14B8A6", "#F59E0B"]

    K = len(excess.columns)
    fig = make_subplots(
        rows=K,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"Rank {i}" for i in range(1, K + 1)],
    )

    for i, c in enumerate(excess.columns, start=1):
        s = excess[c]
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                line=dict(width=2, color=top_colors[(i - 1) % len(top_colors)]),
                name=f"Rank {i}",
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        fig.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="black",
            line_width=1,
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text="Excess return", row=i, col=1)

    fig.update_xaxes(title_text="Date", row=K, col=1)
    fig.update_layout(
        title=title,
        template="plotly",
        height=max(260 * K, 500),
        margin=dict(t=70, b=50, l=80, r=30),
    )

    return fig


def plot_topn_value_and_excess_over_time(
    slot_value_df,
    ret_slot_df,
    passive_ret,
    date_col=None,
    slot_prefix="Top_",
    value_indexed=True,
    title="Value and Excess Return over Time",
    top_colors=None,
):
    def _prep(df, dcol):
        x = df.copy()
        if dcol is not None and dcol in x.columns:
            x.index = pd.to_datetime(x[dcol], errors="coerce")
        else:
            x.index = pd.to_datetime(x.index, errors="coerce")
        x = x.loc[~x.index.isna()].sort_index()
        return x

    def _slot_sort(cols):
        def _k(c):
            s = str(c).split("_", 1)
            return int(s[1]) if len(s) == 2 and s[1].isdigit() else 10**9
        return sorted(cols, key=_k)

    sv0 = _prep(slot_value_df, date_col)
    rs0 = _prep(ret_slot_df, date_col)

    sv_cols = _slot_sort([c for c in sv0.columns if str(c).startswith(slot_prefix)])
    rs_cols = _slot_sort([c for c in rs0.columns if str(c).startswith(slot_prefix)])
    slot_cols = [c for c in sv_cols if c in rs_cols]

    if len(slot_cols) == 0:
        raise ValueError("No common slot columns between slot_value_df and ret_slot_df")

    sv = sv0[slot_cols].apply(pd.to_numeric, errors="coerce")
    rs = rs0[slot_cols].apply(pd.to_numeric, errors="coerce")

    if value_indexed:
        for c in slot_cols:
            s = sv[c]
            fv = s.first_valid_index()
            if fv is None:
                continue
            base = s.loc[fv]
            if pd.notna(base) and base != 0:
                sv[c] = s / base
            else:
                sv[c] = np.nan

    if isinstance(passive_ret, pd.Series):
        p = pd.to_numeric(passive_ret.copy(), errors="coerce")
        p.index = pd.to_datetime(p.index, errors="coerce")
        p = p.loc[~p.index.isna()].sort_index()
    else:
        pr = _prep(passive_ret, date_col)
        cands = [c for c in pr.columns if ("Passive" in str(c)) and (not str(c).endswith("_cumret"))]
        if len(cands) == 0:
            raise KeyError("No raw passive return column found containing 'Passive' (non-cumulative)")
        p = pd.to_numeric(pr[cands[0]], errors="coerce")

    idx_ex = rs.index.intersection(p.index)
    if len(idx_ex) == 0:
        raise ValueError("No overlapping dates between ret_slot_df and passive_ret")

    excess = rs.loc[idx_ex, slot_cols].sub(p.loc[idx_ex], axis=0)

    if top_colors is None:
        top_colors = ["#7C3AED", "#F97316", "#06B6D4", "#0F766E", "#64748B",
                      "#A855F7", "#22C55E", "#EAB308", "#EC4899", "#14B8A6", "#F59E0B"]

    K = len(slot_cols)
    subplot_titles = []
    for i in range(1, K + 1):
        subplot_titles.extend([f"Rank {i} Value", f"Rank {i} Excess Return"])

    fig = make_subplots(
        rows=K,
        cols=2,
        shared_xaxes="columns",
        horizontal_spacing=0.08,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
    )

    for i, c in enumerate(slot_cols, start=1):
        colr = top_colors[(i - 1) % len(top_colors)]

        s_val = sv[c]
        fig.add_trace(
            go.Scatter(
                x=s_val.index,
                y=s_val.values,
                mode="lines",
                line=dict(width=2, color=colr),
                name=f"Rank {i}",
                showlegend=False,
            ),
            row=i,
            col=1,
        )

        s_ex = excess[c]
        fig.add_trace(
            go.Scatter(
                x=s_ex.index,
                y=s_ex.values,
                mode="lines",
                line=dict(width=2, color=colr),
                name=f"Rank {i}",
                showlegend=False,
            ),
            row=i,
            col=2,
        )

        fig.add_hline(
            y=1.0 if value_indexed else 0.0,
            line_dash="dash",
            line_color="black",
            line_width=1,
            row=i,
            col=1,
        )
        fig.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="black",
            line_width=1,
            row=i,
            col=2,
        )

        fig.update_yaxes(title_text="Indexed value" if value_indexed else "Dollar value", row=i, col=1)
        fig.update_yaxes(title_text="Excess return", row=i, col=2)

    fig.update_xaxes(title_text="Date", row=K, col=1)
    fig.update_xaxes(title_text="Date", row=K, col=2)

    fig.update_layout(
        title=title,
        template="plotly",
        height=max(240 * K, 520),
        margin=dict(t=70, b=50, l=70, r=30),
    )

    return fig


def plot_turnover_slippage_skim_info(
    buys_df,
    sells_df,
    core_series,
    cash_alloc_bench=None,
    cash_alloc_rf=None,
    skim_active_col="inj_active_benchmark",
    skim_rfr_col="inj_riskfree_rate",
    skim_total_col="locked_cash_injection",
    title="Turnover, Slippage and Skim Info",
    normalize_dates=True,
):
    def _extract_exec(df):
        if "execution_date" in df.columns:
            ex = pd.to_datetime(df["execution_date"], errors="coerce")
        elif isinstance(df.index, pd.MultiIndex) and ("execution_date" in list(df.index.names)):
            ex = pd.to_datetime(df.index.get_level_values("execution_date"), errors="coerce")
        elif df.index.name == "execution_date":
            ex = pd.to_datetime(df.index, errors="coerce")
        else:
            raise KeyError("execution_date not found as column or index level")
        ex = pd.Series(ex, index=df.index)
        if normalize_dates:
            ex = ex.dt.normalize()
        return ex

    def _count_filled(df, cols):
        z = df[cols].copy()
        z = z.where(~z.isna(), "")
        z = z.astype(str).apply(lambda s: s.str.strip())
        return (z != "").sum(axis=1).astype(float)

    def _prep_side(df, side):
        x = df.copy()
        x["__exec__"] = _extract_exec(x)
        x = x.dropna(subset=["__exec__"])

        if side == "buy":
            n_col = "n_buys"
            prefix = "buy_"
        else:
            n_col = "n_sells"
            prefix = "sell_"

        tcols = [c for c in x.columns if str(c).startswith(prefix)]
        if len(tcols) > 0:
            x[n_col] = _count_filled(x, tcols)
        elif n_col in x.columns:
            x[n_col] = pd.to_numeric(x[n_col], errors="coerce").fillna(0.0)
        else:
            raise KeyError(f"Missing both {n_col} and {prefix}* columns")

        if "cost_bps" in x.columns:
            x["cost_bps"] = pd.to_numeric(x["cost_bps"], errors="coerce").fillna(0.0)
        else:
            x["cost_bps"] = 0.0

        return x.groupby("__exec__", as_index=True)[[n_col, "cost_bps"]].sum().sort_index()

    c = core_series.copy()
    c.index = pd.to_datetime(c.index, errors="coerce")
    c = c.loc[~c.index.isna()].sort_index()
    if normalize_dates:
        c.index = c.index.normalize()
        c = c.groupby(c.index).last()

    if "is_actual_execution_day" in c.columns:
        flag = c["is_actual_execution_day"].fillna(False).astype(bool)
    elif "is_execution_day" in c.columns:
        flag = c["is_execution_day"].fillna(False).astype(bool)
    else:
        raise KeyError("core_series needs is_actual_execution_day or is_execution_day")

    exec_idx = pd.DatetimeIndex(c.index[flag]).unique().sort_values()
    panel = pd.DataFrame(index=exec_idx)

    b = _prep_side(buys_df, "buy").rename(columns={"cost_bps": "buy_cost_bps"})
    s = _prep_side(sells_df, "sell").rename(columns={"cost_bps": "sell_cost_bps"})

    panel = panel.join(b, how="left").join(s, how="left")
    panel["n_buys"] = pd.to_numeric(panel.get("n_buys", 0.0), errors="coerce").fillna(0.0).abs()
    panel["n_sells"] = pd.to_numeric(panel.get("n_sells", 0.0), errors="coerce").fillna(0.0).abs()
    panel["buy_cost_bps"] = pd.to_numeric(panel.get("buy_cost_bps", 0.0), errors="coerce").fillna(0.0)
    panel["sell_cost_bps"] = pd.to_numeric(panel.get("sell_cost_bps", 0.0), errors="coerce").fillna(0.0)
    panel["n_sells_neg"] = -panel["n_sells"]

    if (skim_active_col in c.columns) and (skim_rfr_col in c.columns):
        panel["skim_active"] = pd.to_numeric(c[skim_active_col], errors="coerce").reindex(panel.index).fillna(0.0)
        panel["skim_rfr"] = pd.to_numeric(c[skim_rfr_col], errors="coerce").reindex(panel.index).fillna(0.0)
    else:
        if skim_total_col not in c.columns:
            raise KeyError(f"Need {skim_total_col} or both {skim_active_col} and {skim_rfr_col}")
        if cash_alloc_bench is None or cash_alloc_rf is None:
            raise ValueError("cash_alloc_bench and cash_alloc_rf are required when split skim columns are missing")
        total = pd.to_numeric(c[skim_total_col], errors="coerce").reindex(panel.index).fillna(0.0)
        panel["skim_active"] = total * float(cash_alloc_bench)
        panel["skim_rfr"] = total * float(cash_alloc_rf)

    panel["skim_total"] = panel["skim_active"] + panel["skim_rfr"]
    panel["cum_skim_active"] = panel["skim_active"].cumsum()
    panel["cum_skim_rfr"] = panel["skim_rfr"].cumsum()
    panel["cum_skim_total"] = panel["skim_total"].cumsum()

    panel["slippage_cost_bps"] = panel["buy_cost_bps"] + panel["sell_cost_bps"]
    panel["cum_slippage_cost_bps"] = panel["slippage_cost_bps"].cumsum()

    c_active = "#22C55E"
    c_rfr = "#F59E0B"
    c_total = "#111111"
    c_buys = "#2563EB"
    c_sells = "#DC2626"
    c_slip = "#7C3AED"
    c_slip_cum = "#111827"

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
        specs=[[{}, {}], [{}, {"secondary_y": True}]],
        subplot_titles=[
            "Skim per execution date",
            "Cumulative skim",
            "Turnover per execution date",
            "Slippage per execution date",
        ],
    )

    fig.add_trace(
        go.Bar(x=panel.index, y=panel["skim_active"], name="Skim to Active Benchmark", marker=dict(color=c_active), showlegend=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=panel.index, y=panel["skim_rfr"], name="Skim to RFR", marker=dict(color=c_rfr), showlegend=True),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=panel.index, y=panel["cum_skim_total"], mode="lines", name="Total Skim", line=dict(color=c_total, width=2.6), showlegend=True),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=panel.index, y=panel["cum_skim_active"], mode="lines", name="Skim to Active Benchmark", line=dict(color=c_active, width=2.2), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=panel.index, y=panel["cum_skim_rfr"], mode="lines", name="Skim to RFR", line=dict(color=c_rfr, width=2.2), showlegend=False),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(x=panel.index, y=panel["n_buys"], name="Buys", marker=dict(color=c_buys), showlegend=True),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=panel.index,
            y=-np.abs(panel["n_sells"].values),
            name="Sells",
            marker=dict(color=c_sells),
            customdata=panel["n_sells"].values,
            hovertemplate="Execution=%{x|%Y-%m-%d}<br>Sells=%{customdata}<extra></extra>",
            showlegend=True,
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(x=panel.index, y=panel["slippage_cost_bps"], name="Slippage Cost (bps)", marker=dict(color=c_slip), showlegend=True),
        row=2, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=panel.index,
            y=panel["cum_slippage_cost_bps"],
            mode="lines",
            name="Cumulative Slippage Cost (bps)",
            line=dict(color=c_slip_cum, width=2.6),
            showlegend=True,
        ),
        row=2, col=2, secondary_y=True
    )

    fig.update_layout(
        title=title,
        template="plotly",
        barmode="relative",
        height=900,
        margin=dict(t=70, b=170, l=75, r=75),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0.0, traceorder="normal"),
    )

    fig.update_xaxes(title_text="Execution date", row=1, col=1)
    fig.update_xaxes(title_text="Execution date", row=1, col=2)
    fig.update_xaxes(title_text="Execution date", row=2, col=1)
    fig.update_xaxes(title_text="Execution date", row=2, col=2)

    fig.update_yaxes(title_text="Skim amount", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="Cumulative skim", row=1, col=2, rangemode="tozero")
    fig.update_yaxes(title_text="Turnover count (+ buys / - sells)", row=2, col=1)
    fig.update_yaxes(title_text="Slippage cost per execution (bps)", row=2, col=2, secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Cumulative slippage cost (bps)", row=2, col=2, secondary_y=True, rangemode="tozero")

    m_buy = float(panel["n_buys"].max()) if len(panel) else 0.0
    m_sell = float(panel["n_sells"].max()) if len(panel) else 0.0
    m = int(max(m_buy, m_sell))
    if m < 1:
        m = 1

    tickvals = list(range(-m, m + 1))
    ticktext = [str(abs(v)) for v in tickvals]

    fig.update_yaxes(
        row=2, col=1,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        range=[-m - 0.5, m + 0.5],
        zeroline=True
    )

    fig.update_traces(opacity=1.0, selector=dict(type="bar"))
    fig.update_traces(marker_opacity=1.0, selector=dict(type="bar"))
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))

    return fig, panel


def plot_strategy_performance_relative_to_benchmark(
    opf_df,
    title="Strategy Performance Relative to Benchmark",
    over_col="n_overperformers",
    under_col="n_underperformers",
    excess_col="equity_excess_return",
    drawdown_col="equity_below_notional",
    drawdown_as_negative=True,
):
    d = opf_df.copy()

    if isinstance(d.index, pd.DatetimeIndex):
        x = pd.to_datetime(d.index, errors="coerce")
    elif "period_end" in d.columns:
        x = pd.to_datetime(d["period_end"], errors="coerce")
    else:
        x = pd.to_datetime(d.index, errors="coerce")

    d = d.assign(__x__=x).dropna(subset=["__x__"]).sort_values("__x__").set_index("__x__")

    for c in [over_col, under_col, excess_col, drawdown_col]:
        if c not in d.columns:
            raise KeyError(f"{c} not found in opf_df")

    d[over_col] = pd.to_numeric(d[over_col], errors="coerce").fillna(0.0).abs()
    d[under_col] = pd.to_numeric(d[under_col], errors="coerce").fillna(0.0).abs()
    d[excess_col] = pd.to_numeric(d[excess_col], errors="coerce")
    d[drawdown_col] = pd.to_numeric(d[drawdown_col], errors="coerce").fillna(0.0)

    y_under = -d[under_col].abs()
    y_draw = -d[drawdown_col].abs() if drawdown_as_negative else d[drawdown_col].abs()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Top-N over/underperformers vs benchmark",
            "Equity Sleeve Excess Return",
            "Equity Sleeve Drawdown",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=d.index,
            y=d[over_col],
            name="Overperformers",
            marker=dict(color="#2563EB"),
            opacity=1.0,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=d.index,
            y=y_under,
            name="Underperformers",
            marker=dict(color="#DC2626"),
            customdata=d[under_col].values,
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Underperformers=%{customdata}<extra></extra>",
            opacity=1.0,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d[excess_col],
            mode="lines",
            name="Equity Sleeve Excess Return",
            line=dict(color="#111827", width=2),
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=y_draw,
            mode="lines",
            name="Equity Sleeve Drawdown",
            line=dict(color="#7C3AED", width=2),
            fill="tozeroy",
        ),
        row=3, col=1
    )

    m = int(max(d[over_col].max(), d[under_col].max()))
    if m < 1:
        m = 1
    tickvals = list(range(-m, m + 1))
    ticktext = [str(abs(v)) for v in tickvals]

    fig.update_yaxes(
        row=1, col=1,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        range=[-m - 0.5, m + 0.5],
        zeroline=True,
        title_text="Count (+ over / - under)",
    )

    fig.update_yaxes(row=2, col=1, title_text="Equity Sleeve Excess return", zeroline=True)
    fig.update_yaxes(row=3, col=1, title_text="Equity Sleeve Drawdown", zeroline=True)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    fig.add_hline(y=0, row=2, col=1, line=dict(color="#374151", width=1, dash="dash"))
    fig.add_hline(y=0, row=3, col=1, line=dict(color="#374151", width=1, dash="dash"))

    fig.update_layout(
        title=title,
        template="plotly",
        barmode="relative",
        height=920,
        margin=dict(t=95, b=60, l=80, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )

    return fig


def plot_heatmap_panel(
    agg,
    value="mean",
    top_prefix="Top_",
    gap_rows=1,
    show_text=True,
    text_decimals=2,
    symmetric_scale=True,
    colorscale="RdYlGn",
    title=None,
    top_desc=False,
):
    if value not in ("mean", "count"):
        raise ValueError("value must be 'mean' or 'count'")

    src = agg["mean_table"] if value == "mean" else agg["count_table"]
    cols = list(src.columns)

    def _top_num(c):
        try:
            return int(str(c).split("_")[1])
        except Exception:
            return 10**9

    def _label(c):
        s = str(c)
        if s.startswith(top_prefix):
            try:
                return f"Top {int(s.split('_')[1])}"
            except Exception:
                return s.replace("_", " ")
        return s

    tops = sorted([c for c in cols if str(c).startswith(top_prefix)], key=_top_num, reverse=bool(top_desc))
    eq = "Equity Sleeve" if "Equity Sleeve" in cols else None
    st = "Strategy" if "Strategy" in cols else None
    passive_candidates = [c for c in cols if "passive" in str(c).lower()]
    passive = passive_candidates[0] if len(passive_candidates) > 0 else None

    blocks = []
    if len(tops) > 0:
        blocks.append(tops)
    if eq is not None:
        blocks.append([eq])
    if st is not None:
        blocks.append([st])
    if passive is not None:
        blocks.append([passive])

    used = set([x for b in blocks for x in b])
    extras = [c for c in cols if c not in used]
    if len(extras) > 0:
        blocks.append(extras)

    row_ids, row_text, z_rows = [], [], []
    x_vals = list(src.index)
    g = 0

    for bi, block in enumerate(blocks):
        for c in block:
            row_ids.append(f"row::{c}")
            row_text.append(_label(c))
            z_rows.append(src[c].to_numpy(dtype=float))
        if bi < len(blocks) - 1 and gap_rows > 0:
            for _ in range(gap_rows):
                g += 1
                row_ids.append(f"gap::{g}")
                row_text.append("")
                z_rows.append(np.full(len(x_vals), np.nan))

    z = np.vstack(z_rows)
    finite = z[np.isfinite(z)]
    if len(finite) == 0:
        raise ValueError("No finite values to plot")

    if value == "mean" and symmetric_scale:
        m = float(np.nanmax(np.abs(finite)))
        if m == 0:
            m = 1e-12
        zmin, zmax, zmid = -m, m, 0.0
    else:
        zmin, zmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if zmin == zmax:
            zmin -= 1e-12
            zmax += 1e-12
        zmid = None

    txt = None
    if show_text:
        txt = np.empty(z.shape, dtype=object)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                v = z[i, j]
                if not np.isfinite(v):
                    txt[i, j] = ""
                else:
                    txt[i, j] = f"{100.0*float(v):.{text_decimals}f}%" if value == "mean" else str(int(round(float(v), 0)))

    custom = np.tile(np.array(row_text, dtype=object).reshape(-1, 1), (1, len(x_vals)))

    kwargs = dict(
        z=z,
        x=x_vals,
        y=row_ids,
        customdata=custom,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        zmid=zmid,
        hovertemplate=f"{agg['group_name'].title()}=%{{x}}<br>Series=%{{customdata}}<br>Value=%{{z}}<extra></extra>",
        colorbar=dict(title="Mean" if value == "mean" else "Count"),
        showscale=True,
    )

    if show_text:
        try:
            hm = go.Heatmap(text=txt, texttemplate="%{text}", textfont=dict(size=10), **kwargs)
        except Exception:
            hm = go.Heatmap(text=txt, **kwargs)
    else:
        hm = go.Heatmap(**kwargs)

    fig = go.Figure([hm])

    if title is None:
        title = f"{agg['group_name'].title()} {'Mean Returns' if value == 'mean' else 'Observation Counts'}"

    fig.update_layout(
        title=title,
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1200,
        height=max(360, 42 * len(row_ids) + 180),
    )
    fig.update_xaxes(title_text=agg["group_name"].title(), side="top", showgrid=False, zeroline=False)
    fig.update_yaxes(
        title_text="Series",
        showgrid=False,
        zeroline=False,
        tickmode="array",
        tickvals=row_ids,
        ticktext=row_text,
        categoryorder="array",
        categoryarray=row_ids,
        autorange="reversed",
    )
    return fig


def plot_topn_empirical_and_stats_panel(
    hist,
    slot_col="slot",
    spell_col="spell_id",
    cum_col="cum",
    top_prefix="Top_",
    ncols=3,
    min_obs=3,
    max_h=None,
    max_paths_per_top=None,
    empirical_alpha=0.22,
):
    top_colors = ["#7C3AED", "#F97316", "#06B6D4", "#0F766E", "#64748B",
                  "#A855F7", "#22C55E", "#EAB308", "#EC4899", "#14B8A6", "#F59E0B"]

    d = hist["cum_long"].copy() if isinstance(hist, dict) else hist.copy()
    need = {slot_col, spell_col, cum_col}
    miss = [c for c in need if c not in d.columns]
    if len(miss):
        raise KeyError(f"Missing columns: {miss}")

    d = d.copy()
    d[slot_col] = d[slot_col].astype(str)
    d = d[d[slot_col].str.startswith(top_prefix)].copy()
    d[cum_col] = pd.to_numeric(d[cum_col], errors="coerce")
    d = d.dropna(subset=[cum_col])

    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.sort_values([slot_col, spell_col, "Date"])
    else:
        d = d.sort_values([slot_col, spell_col])

    d["h"] = d.groupby([slot_col, spell_col]).cumcount()
    first_vals = d.groupby([slot_col, spell_col])[cum_col].transform("first")
    d["cum_norm"] = d[cum_col] / first_vals
    d.loc[d["h"] == 0, "cum_norm"] = 1.0

    if max_h is not None:
        d = d[d["h"] <= int(max_h)].copy()

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
    stats["up"] = stats["mean"] + stats["std"]
    stats["dn"] = stats["mean"] - stats["std"]
    stats = stats[stats["n"] >= int(min_obs)].copy()

    def _k(x):
        try:
            return int(str(x).split("_")[1])
        except Exception:
            return 10**9

    slots = sorted(d[slot_col].dropna().unique().tolist(), key=_k)
    if len(slots) == 0:
        raise ValueError("No Top-N slots found to plot")

    def _lbl(x):
        s = str(x)
        if s.startswith("Top_"):
            try:
                return f"Top {int(s.split('_')[1])}"
            except Exception:
                return s.replace("_", " ")
        return s

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(slots) / ncols))
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[_lbl(s) for s in slots],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    shown_emp = False
    shown_mean = False
    shown_std = False
    shown_p = False

    for i, s in enumerate(slots):
        r = i // ncols + 1
        c = i % ncols + 1
        col = top_colors[i % len(top_colors)]

        ds = d[d[slot_col] == s].copy()
        if max_paths_per_top is not None:
            keep_spells = sorted(ds[spell_col].dropna().unique().tolist())[:int(max_paths_per_top)]
            ds = ds[ds[spell_col].isin(keep_spells)]

        for sid, g in ds.groupby(spell_col, sort=True):
            g = g.sort_values("h")
            fig.add_trace(
                go.Scatter(
                    x=g["h"],
                    y=g["cum_norm"],
                    mode="lines",
                    line=dict(color=f"rgba(120,120,120,{float(empirical_alpha)})", width=1),
                    name="Empirical paths",
                    showlegend=not shown_emp,
                    hovertemplate="h=%{x}<br>cum=%{y:.4f}<extra></extra>",
                ),
                row=r,
                col=c,
            )
            shown_emp = True

        zs = stats[stats[slot_col] == s].sort_values("h")
        if len(zs):
            fig.add_trace(
                go.Scatter(
                    x=zs["h"], y=zs["mean"],
                    mode="lines",
                    line=dict(color=col, width=2.8),
                    name="Mean",
                    showlegend=not shown_mean,
                    hovertemplate="h=%{x}<br>mean=%{y:.4f}<extra></extra>",
                ),
                row=r, col=c
            )
            shown_mean = True

            fig.add_trace(
                go.Scatter(
                    x=zs["h"], y=zs["up"],
                    mode="lines",
                    line=dict(color=col, width=1.6, dash="dash"),
                    name="Mean ± 1σ",
                    showlegend=not shown_std,
                    hovertemplate="h=%{x}<br>+1σ=%{y:.4f}<extra></extra>",
                ),
                row=r, col=c
            )
            fig.add_trace(
                go.Scatter(
                    x=zs["h"], y=zs["dn"],
                    mode="lines",
                    line=dict(color=col, width=1.6, dash="dash"),
                    name="Mean ± 1σ",
                    showlegend=False,
                    hovertemplate="h=%{x}<br>-1σ=%{y:.4f}<extra></extra>",
                ),
                row=r, col=c
            )
            shown_std = True

            fig.add_trace(
                go.Scatter(
                    x=zs["h"], y=zs["p10"],
                    mode="lines",
                    line=dict(color=col, width=1.4, dash="dot"),
                    name="p10 / p90",
                    showlegend=not shown_p,
                    hovertemplate="h=%{x}<br>p10=%{y:.4f}<extra></extra>",
                ),
                row=r, col=c
            )
            fig.add_trace(
                go.Scatter(
                    x=zs["h"], y=zs["p90"],
                    mode="lines",
                    line=dict(color=col, width=1.4, dash="dot"),
                    name="p10 / p90",
                    showlegend=False,
                    hovertemplate="h=%{x}<br>p90=%{y:.4f}<extra></extra>",
                ),
                row=r, col=c
            )
            shown_p = True

        fig.update_xaxes(title_text="Day in spell", row=r, col=c)
        fig.update_yaxes(title_text="Cumulative (start=1)", row=r, col=c)

    fig.update_layout(
        template="plotly_white",
        width=460 * ncols,
        height=max(360, 300 * nrows + 120),
        title="Top-N empirical paths with mean, ±1σ, p10/p90",
        legend_title="Lines",
    )
    return fig


def plot_top_stocks_from_ret(
    returns,
    picks_df,
    metadata=None,
    lookbacks=(12, 8, 4),
    topNs=(30, 15, 5),
    skip_t=0,
    date_col=None,
    returns_date_col=None,
    picks_date_col=None,
    execution_col="execution_date",
    top_prefix="Top_",
    background=True,
    title=None,
    tickers_to_plot=None,
    keep_last_row_even_if_unexecuted=True,
):
    def _to_dt_index(df, col_candidates):
        x = df.copy()
        for c in col_candidates:
            if c is not None and c in x.columns:
                idx = pd.to_datetime(x[c], errors="coerce")
                if idx.notna().sum() > 0:
                    x.index = idx
                    x = x.loc[~x.index.isna()].sort_index()
                    return x
        x.index = pd.to_datetime(x.index, errors="coerce")
        x = x.loc[~x.index.isna()].sort_index()
        return x

    if picks_date_col is None:
        picks_date_col = date_col
    if returns_date_col is None:
        returns_date_col = "period_end"

    R0 = _to_dt_index(returns, [returns_date_col, "period_end", None])
    P = _to_dt_index(picks_df, [picks_date_col, "signal_date", None])

    top_cols = [c for c in P.columns if str(c).startswith(top_prefix)]
    if len(top_cols) == 0:
        raise ValueError(f"No columns starting with '{top_prefix}' in picks_df")

    def _top_key(c):
        s = str(c).split("_", 1)
        return int(s[1]) if len(s) == 2 and s[1].isdigit() else 10**9

    top_cols = sorted(top_cols, key=_top_key)

    if metadata is not None:
        md = metadata.copy()
        md.index = pd.to_datetime(md.index, errors="coerce")
        md = md.loc[~md.index.isna()].sort_index()

        if "is_actual_execution_day" in md.columns:
            exec_flag_col = "is_actual_execution_day"
        elif "is_execution_day" in md.columns:
            exec_flag_col = "is_execution_day"
        else:
            raise KeyError("metadata must contain 'is_actual_execution_day' or 'is_execution_day'")

        if execution_col in P.columns:
            ex = pd.to_datetime(P[execution_col], errors="coerce")
            exec_dates = pd.DatetimeIndex(
                md.index[md[exec_flag_col].fillna(False).astype(bool)]
            ).unique()

            mask = ex.isin(exec_dates)
            if keep_last_row_even_if_unexecuted and len(P) > 0:
                max_sig = P.index.max()
                mask = mask | (P.index == max_sig)

            P = P.loc[mask].copy()
        else:
            if "is_actual_signal_day" in md.columns:
                sig_flag_col = "is_actual_signal_day"
            elif "is_signal_day" in md.columns:
                sig_flag_col = "is_signal_day"
            else:
                sig_flag_col = None

            if sig_flag_col is not None:
                sig_dates = pd.DatetimeIndex(
                    md.index[md[sig_flag_col].fillna(False).astype(bool)]
                ).unique()
                P = P.loc[P.index.isin(sig_dates)].copy()

    P = P.loc[P[top_cols].notna().any(axis=1)].copy()
    if P.empty:
        raise ValueError("No eligible signal rows in picks_df after filtering")

    lbs = [int(x) for x in lookbacks]
    Ns = [int(x) for x in topNs]
    if len(Ns) < len(lbs):
        Ns = Ns + [Ns[-1]] * (len(lbs) - len(Ns))
    Ns = Ns[:len(lbs)]

    skip_t = int(skip_t)
    if skip_t < 0:
        raise ValueError("skip_t must be >= 0")

    meta_exact = {
        "weekday_num", "cfg_hold_period",
        "locked_cash", "cash_added_today", "bench_injection_today", "rf_injection_today",
        "current_notional", "equity_value_start", "equity_value_end",
        "equity_return", "stocks_plus_cash", "locked_cash_injection",
        "ret_active_benchmark", "ret_riskfree_rate",
        "into_active_benchmark", "into_riskfree_rate", "into_idle_cash",
        "inj_active_benchmark", "inj_riskfree_rate", "inj_idle_cash",
        "value_active_benchmark", "value_riskfree_rate", "value_idle_cash",
        "value_cash_total", "value_stocks_plus_cash"
    }
    meta_prefixes = ("cfg_", "cal_", "is_", "actual_")
    date_like_cols = {
        "period_start", "period_end",
        "signal_date", "execution_date",
        "execution_date_associated_signal_date",
        "signal_date_associated_with_execution_date",
        "actual_signal_date", "actual_execution_date",
        "actual_execution_date_associated_signal_date",
        "actual_signal_date_associated_with_execution_date"
    }

    num_cols = [c for c in R0.columns if pd.api.types.is_numeric_dtype(R0[c])]
    ret_cols = []
    for c in num_cols:
        sc = str(c)
        if c in meta_exact:
            continue
        if c in date_like_cols:
            continue
        if sc.startswith(meta_prefixes):
            continue
        if sc.lower() in {"weekday", "date"}:
            continue
        ret_cols.append(c)

    if len(ret_cols) == 0:
        raise ValueError("No numeric return columns found in 'returns' after metadata exclusion")

    R = R0[ret_cols].copy().replace([np.inf, -np.inf], np.nan)

    P = P.loc[(P.index >= R.index.min()) & (P.index <= R.index.max())].copy()
    if P.empty:
        raise ValueError("No picks fall inside the returns date range")

    end_signal_dt = P.index.max()
    R_end = R.loc[:end_signal_dt].copy()
    if R_end.empty:
        raise ValueError("No return rows up to selected signal date")

    if skip_t >= len(R_end):
        raise ValueError("skip_t is too large for available history")

    R_rank = R_end.iloc[:-skip_t].copy() if skip_t > 0 else R_end.copy()
    rank_end_dt = R_rank.index.max()

    def _cascade_stage_ranks(Rsig, lbs_, Ns_):
        survivors = list(Rsig.columns)
        out = []
        for lb, N in zip(lbs_, Ns_):
            if len(survivors) == 0 or len(Rsig) < lb:
                out.append({"lb": lb, "N": int(N), "ranks": pd.Series(dtype=float)})
                survivors = []
                continue

            w_stage = Rsig[survivors].tail(lb)
            cum = (1.0 + w_stage).prod(axis=0) - 1.0
            cum = cum.replace([np.inf, -np.inf], np.nan).dropna()

            if cum.empty:
                out.append({"lb": lb, "N": int(N), "ranks": pd.Series(dtype=float)})
                survivors = []
                continue

            order = cum.sort_values(ascending=False).index.tolist()
            rk = pd.Series(np.arange(1, len(order) + 1, dtype=float), index=order)
            out.append({"lb": lb, "N": int(N), "ranks": rk})
            survivors = order[:min(int(N), len(order))]

        return out

    stage_info = _cascade_stage_ranks(R_rank, lbs, Ns)

    if tickers_to_plot is None:
        last_row = P.loc[P.index.max(), top_cols]
        tickers_to_plot = [str(x) for x in last_row.tolist() if pd.notna(x)]
        if len(tickers_to_plot) == 0:
            raise ValueError("Latest Top_* row has no tickers")
    else:
        tickers_to_plot = [str(x) for x in tickers_to_plot]

    missing = [t for t in tickers_to_plot if t not in R_rank.columns]
    if len(missing) > 0:
        raise ValueError(f"Tickers missing in returns: {missing}")

    max_lb = int(max(lbs))
    w = R_rank.tail(min(max_lb, len(R_rank))).copy()
    if w.empty:
        raise ValueError("No return rows available for plotting")

    first_actual_dt = pd.Timestamp(w.index[0])

    base_dt = pd.NaT
    if "period_start" in R0.columns:
        base_val = R0.loc[first_actual_dt, "period_start"] if first_actual_dt in R0.index else pd.NaT
        if isinstance(base_val, pd.Series):
            base_val = base_val.iloc[0]
        base_dt = pd.to_datetime(base_val, errors="coerce")

    if pd.isna(base_dt):
        if len(w.index) >= 2:
            step = w.index[1] - w.index[0]
            if pd.isna(step) or step <= pd.Timedelta(0):
                step = pd.Timedelta(days=1)
        else:
            step = pd.Timedelta(days=1)
        base_dt = first_actual_dt - step

    wealth_actual = (1.0 + w).cumprod()
    wealth_base = pd.DataFrame(1.0, index=pd.DatetimeIndex([base_dt]), columns=wealth_actual.columns)
    data_all = pd.concat([wealth_base, wealth_actual], axis=0)
    data_to_plot = data_all[tickers_to_plot].copy()

    def _rank_label(tk):
        parts = []
        for st in stage_info:
            lb = st["lb"]
            N = st["N"]
            rs = st["ranks"]
            v = rs.get(tk, np.nan)
            parts.append(f"r{lb}:{int(v)}/{int(N)}" if np.isfinite(v) else f"r{lb}:NA/{int(N)}")
        return " | ".join(parts)

    subplot_titles = [
        f"{tk}<br><span style='font-size:10px'>{_rank_label(tk)}</span>"
        for tk in tickers_to_plot
    ]
    fig = make_subplots(rows=1, cols=len(tickers_to_plot), subplot_titles=subplot_titles)

    TOP_COLORS = [
        "#7C3AED", "#F97316", "#06B6D4", "#0F766E", "#64748B",
        "#A855F7", "#22C55E", "#EAB308", "#EC4899", "#14B8A6", "#F59E0B"
    ]

    if background:
        x_bg, y_bg = [], []
        x_base = list(data_all.index)
        for c0 in data_all.columns:
            y = data_all[c0].values.tolist()
            x_bg.extend(x_base + [None])
            y_bg.extend(y + [None])

    lb_lines = [w.index[-lb] for lb in lbs if lb <= len(w.index)]

    for i, tk in enumerate(tickers_to_plot):
        series = data_to_plot[tk]
        r0, c0 = 1, i + 1

        if background:
            fig.add_trace(
                go.Scatter(
                    x=x_bg,
                    y=y_bg,
                    mode="lines",
                    line=dict(color="rgba(120,120,120,0.18)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=r0, col=c0
            )

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line=dict(width=3, color=TOP_COLORS[i % len(TOP_COLORS)]),
                showlegend=False,
            ),
            row=r0, col=c0
        )

        vals = series.values
        finite = np.isfinite(vals)
        ymin = float(np.nanmin(vals[finite])) if finite.any() else 0.95
        ymax = float(np.nanmax(vals[finite])) if finite.any() else 1.05

        for xline in lb_lines:
            fig.add_shape(
                type="line",
                x0=xline,
                x1=xline,
                y0=ymin,
                y1=ymax,
                line=dict(color="black", dash="dash"),
                row=r0,
                col=c0
            )

    if title is None:
        lb_txt = ", ".join(str(x) for x in lbs)
        title = f"Cumulative returns over the last {lb_txt} lookback periods (ending {pd.to_datetime(rank_end_dt).date()})"

    fig.update_layout(
        title=title,
        height=330,
        width=max(1200, 240 * len(tickers_to_plot)),
        template="plotly_white",
        showlegend=False,
        margin=dict(t=80, b=50, l=40, r=20),
    )

    return fig


def plot_topn_clean_drilldown(
    picks_df,
    metadata=None,
    date_col="signal_extraction_date",
    top_prefix="Top_",
    res=None,
    trades_log=None,
    execution_col="execution_date",
    hold_period=1,
    title=None,
    meta_exec_flag_col=None,
    meta_signal_flag_col=None,
):
    df0 = picks_df.copy()

    top_cols = [c for c in df0.columns if str(c).startswith(top_prefix)]
    if not top_cols:
        raise ValueError(f"No columns starting with '{top_prefix}' in picks_df")

    def _top_key(c):
        try:
            return int(str(c).split("_")[1])
        except Exception:
            return 10**9

    top_cols = sorted(top_cols, key=_top_key)

    if date_col in df0.columns:
        dt = pd.to_datetime(df0[date_col])
    else:
        dt = pd.to_datetime(df0.index)

    if metadata is not None:
        md = metadata.copy()
        md.index = pd.to_datetime(md.index)
        md = md.sort_index()

        if meta_exec_flag_col is None:
            if "is_actual_execution_day" in md.columns:
                meta_exec_flag_col = "is_actual_execution_day"
            elif "is_execution_day" in md.columns:
                meta_exec_flag_col = "is_execution_day"
            else:
                raise KeyError("metadata must contain 'is_actual_execution_day' or 'is_execution_day'")

        if meta_signal_flag_col is None:
            if "is_signal_day" in md.columns:
                meta_signal_flag_col = "is_signal_day"
            elif "is_actual_signal_day" in md.columns:
                meta_signal_flag_col = "is_actual_signal_day"
            else:
                meta_signal_flag_col = None

        exec_dates = pd.DatetimeIndex(md.index[md[meta_exec_flag_col].fillna(False).astype(bool)]).unique().sort_values()
        if len(exec_dates) == 0:
            raise ValueError("No execution dates flagged in metadata")

        sig_dates = pd.DatetimeIndex([])
        if meta_signal_flag_col is not None:
            sig_dates = pd.DatetimeIndex(md.index[md[meta_signal_flag_col].fillna(False).astype(bool)]).unique().sort_values()

        if execution_col in df0.columns:
            ex = pd.to_datetime(df0[execution_col])
            keep = ex.isin(exec_dates)
            if len(sig_dates) > 0:
                keep = keep | dt.isin(sig_dates)
        else:
            if meta_signal_flag_col is None:
                raise KeyError(f"'{execution_col}' not in picks_df and no signal flag column found in metadata")
            keep = dt.isin(sig_dates)

        df_f = df0.loc[keep].copy()

    else:
        if trades_log is None and isinstance(res, dict):
            trades_log = res.get("trades_log", None)

        if trades_log is not None:
            tl = trades_log.copy()
            tl.index = pd.to_datetime(tl.index)
            rb = pd.DatetimeIndex(tl.index).sort_values().unique()

            if execution_col in df0.columns:
                ex = pd.to_datetime(df0[execution_col])
                keep = ex.isin(rb)
            else:
                keep = dt.isin(rb)

            df_f = df0.loc[keep].copy()
        else:
            hold_period = int(hold_period)
            if hold_period < 1:
                raise ValueError("hold_period must be >= 1")

            if hold_period == 1:
                df_f = df0.copy()
            else:
                tmp = df0[top_cols].copy()
                tmp.index = pd.to_datetime(dt.values)
                tmp = tmp.sort_index()

                ok = tmp.notna().any(axis=1)
                if not ok.any():
                    raise ValueError("No non-NA picks to plot")

                idx = tmp.index
                start_i = int(np.argmax(ok.values))
                keep_dates = idx[start_i::hold_period]
                keep = pd.to_datetime(dt).isin(pd.to_datetime(pd.Index(keep_dates)))
                df_f = df0.loc[keep].copy()

    dfp = df_f.copy()
    if date_col in dfp.columns:
        dfp[date_col] = pd.to_datetime(dfp[date_col])
        dfp = dfp.set_index(date_col)
    else:
        dfp.index = pd.to_datetime(dfp.index)

    dfp = dfp.sort_index()
    top = dfp[top_cols].copy().dropna(how="all")
    if top.empty:
        raise ValueError("No picks to plot after filtering")

    K = len(top.columns)

    long = top.reset_index().rename(columns={top.index.name or "index": "Date"})
    long = long.melt(id_vars=["Date"], var_name="Rank", value_name="Ticker")
    long = long.dropna(subset=["Ticker"])
    long["Ticker"] = long["Ticker"].astype(str)
    long["Rank_num"] = long["Rank"].str.extract(r"(\d+)").astype(int)
    long["Year"] = long["Date"].dt.year

    years = sorted(long["Year"].unique().tolist())
    if not years:
        raise ValueError("No picks to plot")

    tickers = sorted(long["Ticker"].unique().tolist())
    palette = (
        px.colors.qualitative.Alphabet
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Pastel
    )
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(tickers)}

    fig = go.Figure()
    traces_by_year = {}
    dates_by_year = {}

    for y in years:
        dy = long[long["Year"] == y].copy()
        dates = sorted(dy["Date"].unique().tolist())
        dates_by_year[y] = dates
        traces_by_year[y] = []

        for rnk in range(K, 0, -1):
            dr = dy[dy["Rank_num"] == rnk].set_index("Date").reindex(dates)
            tkr = dr["Ticker"].astype(object).where(dr["Ticker"].notna(), "")
            colors = [color_map.get(v, "#999999") if v else "rgba(0,0,0,0)" for v in tkr.tolist()]
            yvals = [1.0 if v else 0.0 for v in tkr.tolist()]

            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=yvals,
                    marker=dict(color=colors, line=dict(width=0)),
                    text=tkr.tolist(),
                    texttemplate="%{text}",
                    textposition="inside",
                    textangle=90,
                    insidetextanchor="middle",
                    textfont=dict(size=7),
                    constraintext="none",
                    cliponaxis=False,
                    hovertemplate="Date=%{x|%Y-%m-%d}<br>Rank=" + str(rnk) + "<br>Ticker=%{text}<extra></extra>",
                    visible=(y == years[0]),
                    showlegend=False,
                )
            )
            traces_by_year[y].append(len(fig.data) - 1)

    def _ticktext(dates):
        return [pd.to_datetime(x).strftime("%Y-%m-%d") for x in dates]

    buttons = []
    for y in years:
        vis = [False] * len(fig.data)
        for i in traces_by_year[y]:
            vis[i] = True
        buttons.append(
            dict(
                label=str(y),
                method="update",
                args=[
                    {"visible": vis},
                    {
                        "title": f"Top-{K} drilldown ({y})",
                        "xaxis": {
                            "tickmode": "array",
                            "tickvals": dates_by_year[y],
                            "ticktext": _ticktext(dates_by_year[y]),
                        },
                    },
                ],
            )
        )

    tickvals = [i + 0.5 for i in range(K)]
    ticktext = [str(v) for v in range(K, 0, -1)]

    y0 = years[0]
    fig.update_layout(
        barmode="stack",
        title=f"Top-{K} drilldown ({y0})",
        template="plotly_white",
        height=520,
        margin=dict(t=70, b=50, l=80, r=40),
        updatemenus=[dict(type="dropdown", buttons=buttons, x=1.02, y=1.0)],
        uniformtext_minsize=6,
        uniformtext_mode="show",
        bargap=0.05,
    )
    fig.update_yaxes(
        range=[0, K],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        title="Rank (1 = best)",
        showgrid=False,
        zeroline=False,
    )
    fig.update_xaxes(
        title="Signal extraction date",
        showgrid=False,
        tickmode="array",
        tickvals=dates_by_year[y0],
        ticktext=_ticktext(dates_by_year[y0]),
    )

    if title is not None:
        fig.update_layout(title=title)

    return fig
