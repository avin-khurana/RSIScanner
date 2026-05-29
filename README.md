# RSI Oversold + LEAPS IV Scanner

Scans the top 40 large-cap US stocks (market cap > $100B) for RSI(14) oversold conditions, then surfaces a matching LEAPS call with real implied volatility, greeks, and IV rank context.

## What it does

1. **Universe** — fetches the S&P 100 constituent list from Wikipedia; falls back to a hardcoded large-cap list. Filters to the top 40 by market cap > $100B.
2. **RSI signal** — fires if RSI(14) was below 30 on *any* of the last 3 trading days (catches signals that already started reverting).
3. **LEAPS contract** — finds the nearest expiry ≥ 360 days out, selects the ITM call closest to Δ = 0.70.
4. **Real IV** — back-solves implied volatility from the live bid/ask mid via Newton-Raphson (Black-Scholes inversion) rather than trusting Yahoo Finance's pre-computed `impliedVolatility` field, which is stale (computed from last trade) and often NaN or 0 for deep ITM LEAPS.
5. **IV Rank** — compares current 30-day realized vol against its 52-week range and labels it with a tiered buy guidance.
6. **Email report** — sends a rich HTML summary to a configured address.

## IV Rank tiers

| Rank | Label | Guidance |
|------|-------|----------|
| < 20 | **IDEAL BUY** | IV near 52-wk low — cheapest options of the year |
| 20–40 | **GOOD** | IV below average — favorable for buying |
| 40–60 | **FAIR** | IV near average — neutral entry |
| 60–80 | **ABOVE AVG** | IV above average — consider spreads over naked longs |
| > 80 | **EXPENSIVE** | IV near 52-wk high — poor time to buy naked options |

## How implied volatility is resolved

The scanner uses a three-tier fallback for each LEAPS contract:

1. **Back-solved IV** (primary) — Newton-Raphson solver finds the σ that makes `BS_call(σ) = bid_ask_mid`. Converges in ~5–10 iterations; accurate to floating-point precision.
2. **Yahoo's `impliedVolatility`** field — used only if the solver returns `None` (e.g. zero-width bid/ask after hours).
3. **30-day realized volatility** — last resort when neither chain source is usable.

This matters because Yahoo's pre-computed IV for deep ITM LEAPS can diverge from the real market IV by several percentage points (observed: 34.7% Yahoo vs 27.0% solved for the same contract).

## Setup

```bash
pip install -r requirements.txt
```

Environment variables for email (optional — script prints results to stdout regardless):

```bash
export EMAIL_FROM="you@gmail.com"
export EMAIL_APP_PASSWORD="your-app-password"
export EMAIL_TO="you@gmail.com"          # defaults to the address above
```

## Usage

```bash
python rsi_scanner.py
```

Results are also saved to `rsi_scan_results.csv`.

## Key configuration

```python
MIN_MARKET_CAP = 100e9    # $100B minimum market cap
TOP_N          = 40       # number of tickers to scan
RSI_PERIOD     = 14       # RSI period
RSI_THRESHOLD  = 30.0     # oversold threshold
RSI_LOOKBACK   = 3        # signal fires if RSI < threshold on any of the last N days
TARGET_DELTA   = 0.70     # ITM call delta target for LEAPS
MIN_LEAPS_DAYS = 360      # minimum days to expiry
RISK_FREE_RATE = 0.045    # used in Black-Scholes
```

## RSI implementation

Uses Wilder's smoothing (`EMA with com = period - 1`) to match TradingView's RSI exactly, not the simple rolling-average approximation used by many libraries.

## Output example

```
WMT    RSI=[29.1 / 33.0 / 32.0]  Price=$115.86  IV Rank=96/100 [EXPENSIVE]
       IV guidance: IV near 52-wk high — poor time to buy naked options
       IV scale: <20 IDEAL BUY · 20-40 GOOD · 40-60 FAIR · 60-80 ABOVE AVG · >80 EXPENSIVE
       LEAPS: $105 Call (2027-06-17)  IV=27%  Rank=96 [EXPENSIVE]  Prem=$21.35  BE=+9.1%
```

## Automated runs

A GitHub Actions workflow (`.github/workflows/`) can be configured to run the scanner on a schedule and deliver the email report daily before market open.

## Disclaimer

Not financial advice. Data sourced from Yahoo Finance via `yfinance`. Options data may be delayed or incomplete outside market hours.
