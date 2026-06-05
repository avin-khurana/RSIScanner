# RSI Oversold + LEAPS IV Scanner

Scans the top 40 large-cap US stocks (market cap > $100B) for RSI(14) oversold conditions, then surfaces a matching LEAPS call with real implied volatility, greeks, and IV rank context.

## What it does

1. **Universe** — fetches the S&P 100 constituent list from Wikipedia; falls back to a hardcoded large-cap list. Filters to the top 40 by market cap > $100B.
2. **RSI signal** — fires if RSI(14) was below 30 on *any* of the last 3 trading days (catches signals that already started reverting).
3. **LEAPS contract** — finds the nearest expiry ≥ 360 days out, selects the ITM call closest to Δ = 0.70.
4. **Real IV** — back-solves implied volatility from the live bid/ask mid via Newton-Raphson (Black-Scholes inversion) rather than trusting Yahoo Finance's pre-computed `impliedVolatility` field, which is stale (computed from last trade) and often NaN or 0 for deep ITM LEAPS.
5. **IV Rank** — compares current 30-day realized vol against its 52-week range and labels it with a tiered buy guidance.
6. **Vega-gain estimate** — shows the estimated % option gain if IV reverts from its current level back to the 52-week average (useful for spotting depressed-IV setups).
7. **Trade management levels** — for each signal, back-solves and prints:
   - **Stock target** (▲): the stock price at which the LEAPS call gains +50%, computed via Black-Scholes binary search (accounts for delta, gamma, and remaining theta).
   - **Stock stop** (▼): entry stock × 0.90 — a consistent 10% stock-level stop that gives every position the same breathing room regardless of option premium size.
   - A ready-to-paste CSV row for the trade tracker.
8. **Near-misses watch list** — the email also includes a table of stocks with RSI 30–35 (not yet oversold, but worth monitoring).
9. **Email report** — sends a rich HTML summary to a configured address, including per-ticker signal cards with greeks (Δ, Θ, Vega), target/stop levels, and the near-misses table.

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
export EMAIL_TO="you@gmail.com"          # optional; defaults to avin.khurana18@gmail.com if unset
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
WMT    RSI(last 3d)=[29.1 / 33.0 / 32.0]  Price=$115.86  IV Rank=96/100 [EXPENSIVE]
       IV guidance: IV near 52-wk high — poor time to buy naked options
       IV scale: <20 IDEAL BUY · 20-40 GOOD · 40-60 FAIR · 60-80 ABOVE AVG · >80 EXPENSIVE
       LEAPS: $105 Call (2027-06-17)  IV=27%  Rank=96 [EXPENSIVE]  Prem=$21.35  BE=+9.1%
       Target : stock ≥ $128.40 (+10.8%)  →  option +50% ($32.03)
       Stop   : stock ≤ $104.27 (−10.0%)  →  10% stock-level stop
       ┌─ Paste into my_trades.csv ───────────────────────────────────
       │ WMT,2026-06-05,115.86,105,C,2027-06-17,21.35,128.40,104.27,32.03,31,OPEN,,,,,
       └──────────────────────────────────────────────────────────────
```

The HTML email adds per-ticker greeks (Δ, Θ/day, Vega/1%IV), the IV 52-week range (low → avg → high), vega-gain estimate, and a Target ▲ / Stop ▼ row with the back-solved stock price levels. Near-misses table (RSI 30–35) is also included.

## Trade Tracker

`trade_tracker.py` monitors your open LEAPS positions against live stock prices and sends alerts when a target or stop is crossed.

### Workflow

1. Run the scanner — it prints a ready-to-paste CSV row for each signal.
2. Add your **actual fills** to `my_trades.csv` (copy the row, adjust entry prices to match your broker).
3. Run the tracker to check status:

```bash
python trade_tracker.py           # check all positions, fire alerts if target/stop hit
python trade_tracker.py --report  # save trade_report.html and open it
python trade_tracker.py --summary # print table only, no alerts
```

4. When you close a trade, set `status=WIN` or `status=LOSS` and fill in the exit columns.
5. `git add my_trades.csv && git commit` to version-control your trade log.

### Alerts

- **macOS desktop notification** — pops immediately when target or stop is crossed.
- **Email** — sends a formatted alert card to your configured `EMAIL_TO`.
- No duplicate alerts: re-running the tracker after an alert does not resend.

### my_trades.csv columns

| Column | Description |
|--------|-------------|
| `ticker` | Stock symbol |
| `entry_date` | Date you entered (YYYY-MM-DD) |
| `entry_stock` | Your actual fill price on the stock |
| `strike` | Option strike |
| `option_type` | C or P |
| `expiry` | Option expiry (YYYY-MM-DD) |
| `entry_option` | Premium you actually paid per share |
| `target_stock` | Stock price for +50% option gain (from scanner) |
| `stop_stock` | 10% below entry stock (from scanner) |
| `target_option` | Option price at +50% gain |
| `sigma_pct` | IV% at entry — used for live BS repricing |
| `status` | OPEN / WIN / LOSS |
| `exit_date / exit_stock / exit_option / pnl_pct` | Fill when closing |
| `notes` | Anything worth remembering |

## Automated runs

A GitHub Actions workflow (`.github/workflows/`) can be configured to run the scanner on a schedule and deliver the email report daily before market open.

## Disclaimer

Not financial advice. Data sourced from Yahoo Finance via `yfinance`. Options data may be delayed or incomplete outside market hours.
