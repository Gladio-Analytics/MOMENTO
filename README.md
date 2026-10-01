# MOMENTR

Cross-sectional momentum backtest dashboard, built on Streamlit.

## Setup

```
pip install -r requirements.txt
```

Build the local price cache once (reads `Data Input/prices.parquet`, used by the app for the US/CAN universes):

```
jupyter notebook UPDATE_DB.ipynb
```

Run all cells to fetch and cache index constituents' prices.

## Run

```
streamlit run momentr_app.py
```

## Layout

- `momentr_app.py` - the dashboard
- `momentum_backend.py` - signal construction, portfolio ledger, cash/notional accounting
- `momentum_fetch.py` - price data fetch/cache helpers
- `momentum_viz.py` - Plotly chart functions
- `UPDATE_DB.ipynb` - builds the local price cache the app reads from
