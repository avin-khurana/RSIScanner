#!/usr/bin/env python3
"""
RSI Oversold LEAPS Call Backtest — Long side
Signal : RSI(14) < 30 on any of last 3 trading days (first occurrence per ticker only)
Filter : --iv-rank-max N  (optional) rejects entries where IV Rank ≥ N
Stop   : --stock-stop 0.10  exits when STOCK falls X% from entry (stock-level stop, recommended)
         default is -30% on option price (original behaviour)
Exit   : +50% option gain OR stop triggered; else shown as OPEN with current estimated P&L

Usage:
  python rsi_backtest_long.py                            # 2-year, Δ=0.70, option-level -30% stop
  python rsi_backtest_long.py --stock-stop 0.10          # 10% underlying stock stop (recommended)
  python rsi_backtest_long.py --years 3 --stock-stop 0.10
  python rsi_backtest_long.py --iv-rank-max 40           # IV rank filter (shown to be counterproductive)
"""

import argparse
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
RSI_PERIOD     = 14
RSI_THRESHOLD  = 30.0
RSI_LOOKBACK   = 3
MIN_LEAPS_DAYS = 360
RISK_FREE_RATE = 0.045
PROFIT_TARGET  = 0.50   # +50%
LOSS_TARGET    = -0.30  # −30%

UNIVERSE = [
    'AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','BRK-B',
    'UNH','XOM','JPM','JNJ','V','PG','MA','HD','CVX','MRK','ABBV',
    'LLY','COST','PEP','KO','AVGO','BAC','WMT','MCD','CRM','NFLX',
    'TMO','AMD','ACN','NEE','ORCL','QCOM','TXN','IBM','GS','SPGI',
    'LIN','CAT','BLK','AXP','GE','AMGN','HON','RTX','PM','DHR',
][:40]


# ── Indicators ────────────────────────────────────────────────────────────────
def fix_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def calc_rsi(closes, period=RSI_PERIOD):
    delta    = closes.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))

def realized_vol(closes, window=30):
    return np.log(closes / closes.shift(1)).dropna().rolling(window).std() * np.sqrt(252)

def iv_rank_at_entry(rv_series, lookback=252):
    """IV Rank: where current 30-day realized vol sits in its 52-week range (0–100).
    Identical logic to rsi_scanner.py iv_stats(). Returns (rank, lo_pct, hi_pct, avg_pct)."""
    clean = rv_series.dropna()
    if len(clean) < 30:
        return 50.0, 0.0, 0.0, 0.0
    period = clean.iloc[-lookback:]
    lo  = float(period.min())
    hi  = float(period.max())
    avg = float(period.mean())
    cur = float(clean.iloc[-1])
    rank = round((cur - lo) / (hi - lo) * 100, 1) if hi > lo else 50.0
    return rank, round(lo * 100, 1), round(hi * 100, 1), round(avg * 100, 1)


# ── Black-Scholes ─────────────────────────────────────────────────────────────
def bs_greeks(S, K, T, r=RISK_FREE_RATE, sigma=0.30):
    if T <= 1e-6 or S <= 0 or K <= 0 or sigma <= 0:
        return None
    d1  = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2  = d1 - sigma * np.sqrt(T)
    pdf = norm.pdf(d1)
    return dict(
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
        delta = norm.cdf(d1),
        theta = (-(S * pdf * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365,
        vega  = S * pdf * np.sqrt(T) / 100,
    )

def find_call_strike(S, target_delta, T, r, sigma):
    """Binary search for ITM call strike where call delta ≈ target_delta."""
    lo, hi = S * 0.4, S * 1.8
    mid = S
    for _ in range(60):
        mid = (lo + hi) / 2
        g = bs_greeks(S, mid, T, r, sigma)
        if g is None:
            break
        if g['delta'] > target_delta:
            lo = mid
        else:
            hi = mid
    return round(mid / 5) * 5


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(years, target_delta, iv_rank_max=None, stock_stop=None):
    today = datetime.now().date()
    start = today - timedelta(days=int(years * 365.25))

    # Download 5y so RSI has enough lookback even at the start of a 3-year window
    print(f"\nDownloading 5y price history for {len(UNIVERSE)} tickers...")
    price_data = {}
    for ticker in UNIVERSE:
        try:
            df = fix_cols(yf.download(ticker, period='5y', progress=False, auto_adjust=True))
            if len(df) >= 60:
                price_data[ticker] = df['Close'].squeeze()
                print(f"  {ticker:<6} OK ({len(df)} days)")
            else:
                print(f"  {ticker:<6} skipped — insufficient history")
        except Exception as e:
            print(f"  {ticker:<6} error: {e}")

    all_dates = sorted({
        idx.date()
        for series in price_data.values()
        for idx in series.index
        if start <= idx.date() <= today
    })

    iv_filter_label  = f"IV Rank < {iv_rank_max}" if iv_rank_max is not None else "none"
    stop_label       = f"{stock_stop*100:.0f}% stock drop" if stock_stop else f"-30% option loss"
    print(f"\nScanning {len(all_dates)} trading days ({start} → {today})  "
          f"[{years}-year window, Δ≈{target_delta}, stop: {stop_label}, IV filter: {iv_filter_label}]...")

    trades       = []
    triggered    = set()
    filtered_out = []   # signals that fired but failed the IV rank filter

    for d in all_dates:
        for ticker, closes in price_data.items():
            if ticker in triggered:
                continue

            closes_to_d = closes[closes.index.date <= d]
            if len(closes_to_d) < RSI_PERIOD + RSI_LOOKBACK + 5:
                continue

            rsi_series = calc_rsi(closes_to_d).dropna()
            if len(rsi_series) < RSI_LOOKBACK:
                continue

            rsi_last = [
                round(float(rsi_series.iloc[-i]), 1) if len(rsi_series) >= i else None
                for i in range(1, RSI_LOOKBACK + 1)
            ]

            if not any(v is not None and v < RSI_THRESHOLD for v in rsi_last):
                continue

            # RSI signal fired — now compute IV rank before deciding to enter
            S     = float(closes_to_d.iloc[-1])
            rv    = realized_vol(closes_to_d, 30)
            sigma = float(rv.dropna().iloc[-1]) if not rv.dropna().empty else 0.30
            rank, iv_lo, iv_hi, iv_avg = iv_rank_at_entry(rv)

            triggered.add(ticker)  # mark triggered regardless — first signal only

            if iv_rank_max is not None and rank >= iv_rank_max:
                print(f"  {ticker:<6} FILTERED {d}  RSI={rsi_last[0]:.1f}  "
                      f"IV Rank={rank:.0f} ≥ {iv_rank_max} → skipped")
                filtered_out.append(dict(ticker=ticker, date=d, rsi=rsi_last[0], iv_rank=rank))
                continue

            T_entry = MIN_LEAPS_DAYS / 365
            K       = find_call_strike(S, target_delta, T_entry, RISK_FREE_RATE, sigma)
            g_entry = bs_greeks(S, K, T_entry, RISK_FREE_RATE, sigma)
            if g_entry is None:
                continue

            trades.append(dict(
                ticker      = ticker,
                entry_date  = d,
                entry_stock = round(S, 2),
                strike      = K,
                T_entry     = T_entry,
                sigma       = sigma,
                iv_rank     = rank,
                iv_lo       = iv_lo,
                iv_hi       = iv_hi,
                iv_avg      = iv_avg,
                entry_price = round(g_entry['price'], 2),
                entry_delta = round(g_entry['delta'], 2),
                rsi_days    = rsi_last,
                exit_date   = None,
                exit_stock  = None,
                exit_price  = None,
                pnl_pct     = None,
                outcome     = 'OPEN',
            ))
            print(f"  {ticker:<6} signal {d}  RSI={rsi_last[0]:.1f}  "
                  f"IV Rank={rank:.0f}  S=${S:.2f}  K=${K:.0f}  "
                  f"IV={sigma*100:.0f}%  entry=${g_entry['price']:.2f}")

    print(f"\nFound {len(trades)} signal(s) "
          f"({len(filtered_out)} filtered by IV Rank). Simulating exits...")

    for trade in trades:
        ticker   = trade['ticker']
        closes   = price_data[ticker]
        K, sigma, T_entry, ep = trade['strike'], trade['sigma'], trade['T_entry'], trade['entry_price']
        entry_d  = trade['entry_date']
        S_entry  = trade['entry_stock']

        for idx, price_val in closes[closes.index.date > entry_d].items():
            days_elapsed = (idx.date() - entry_d).days
            T_now = T_entry - days_elapsed / 365
            if T_now <= 0.01:
                break
            S_now = float(price_val)
            g = bs_greeks(S_now, K, T_now, RISK_FREE_RATE, sigma)
            if g is None:
                continue
            opt_pnl = (g['price'] - ep) / ep

            # Stock-level stop: consistent $ risk regardless of option premium size
            if stock_stop is not None:
                stopped = (S_now - S_entry) / S_entry <= -stock_stop
            else:
                stopped = opt_pnl <= LOSS_TARGET

            if opt_pnl >= PROFIT_TARGET:
                trade.update(exit_date=idx.date(), exit_stock=round(S_now, 2),
                             exit_price=round(g['price'], 2), pnl_pct=round(opt_pnl * 100, 1), outcome='WIN')
                break
            elif stopped:
                trade.update(exit_date=idx.date(), exit_stock=round(S_now, 2),
                             exit_price=round(g['price'], 2), pnl_pct=round(opt_pnl * 100, 1), outcome='LOSS')
                break

        if trade['outcome'] == 'OPEN':
            last = closes[closes.index.date <= today]
            if not last.empty:
                S_now = float(last.iloc[-1])
                T_now = max(T_entry - (today - entry_d).days / 365, 0.01)
                g = bs_greeks(S_now, K, T_now, RISK_FREE_RATE, sigma)
                if g:
                    pnl = (g['price'] - ep) / ep
                    trade.update(exit_date=today, exit_stock=round(S_now, 2),
                                 exit_price=round(g['price'], 2), pnl_pct=round(pnl * 100, 1))

    return trades, filtered_out


# ── HTML ──────────────────────────────────────────────────────────────────────
def generate_html(trades, filtered_out, years, target_delta, iv_rank_max, stock_stop, output_path):
    today_str = datetime.now().strftime('%Y-%m-%d')

    wins   = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']
    opens  = [t for t in trades if t['outcome'] == 'OPEN']
    closed = wins + losses

    win_rate    = len(wins) / len(closed) * 100 if closed else 0
    avg_pnl_cl  = np.mean([t['pnl_pct'] for t in closed]) if closed else 0
    avg_pnl_all = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] is not None]) if trades else 0

    # Best / worst
    all_closed_pnl = sorted([t['pnl_pct'] for t in closed if t['pnl_pct'] is not None], reverse=True)
    best_pnl  = all_closed_pnl[0]  if all_closed_pnl else None
    worst_pnl = all_closed_pnl[-1] if all_closed_pnl else None

    # Avg hold days (closed only)
    hold_days_list = [
        (t['exit_date'] - t['entry_date']).days
        for t in closed if t.get('exit_date')
    ]
    avg_hold = round(np.mean(hold_days_list)) if hold_days_list else 0

    sorted_trades = (
        sorted(wins,   key=lambda x: -(x['pnl_pct'] or 0)) +
        sorted(opens,  key=lambda x: -(x['pnl_pct'] or 0)) +
        sorted(losses, key=lambda x: -(x['pnl_pct'] or 0))
    )

    rows = ""
    for t in sorted_trades:
        outcome = t['outcome']
        color      = {'WIN': '#3fb950', 'LOSS': '#f85149', 'OPEN': '#d29922'}[outcome]
        row_border = {'WIN': 'border-left:3px solid #3fb950',
                      'LOSS': 'border-left:3px solid #f85149',
                      'OPEN': 'border-left:3px solid #d29922'}[outcome]
        badge_bg   = {'WIN': '#1a3a1a', 'LOSS': '#3a1a1a', 'OPEN': '#2a2510'}[outcome]
        badge = (f'<span style="background:{badge_bg};color:{color};padding:3px 12px;'
                 f'border-radius:10px;font-size:12px;font-weight:bold">{outcome}</span>')

        rsi_cells = ' / '.join(
            (f'<span style="color:{("#f85149" if v < RSI_THRESHOLD else "#c9d1d9")}">'
             f'{"★ " if v < RSI_THRESHOLD else ""}{v:.1f}</span>')
            if v is not None else '<span style="color:#8b949e">—</span>'
            for v in t.get('rsi_days', [])
        )
        hold = (t['exit_date'] - t['entry_date']).days if t.get('exit_date') else '—'
        pnl_str = (f'<span style="color:{color};font-weight:bold">{t["pnl_pct"]:+.1f}%</span>'
                   if t['pnl_pct'] is not None else '—')
        ep_str   = f"${t['exit_price']:.2f}" if t.get('exit_price') else '—'
        est_note = ' <span style="color:#8b949e;font-size:10px">(est.)</span>' if outcome == 'OPEN' else ''

        rank = t.get('iv_rank', None)
        if rank is not None:
            if rank < 20:   rc = '#3fb950'
            elif rank < 40: rc = '#56d364'
            elif rank < 60: rc = '#d29922'
            elif rank < 80: rc = '#e3a341'
            else:           rc = '#f85149'
            rank_cell = f'<span style="color:{rc};font-weight:bold">{rank:.0f}</span>'
        else:
            rank_cell = '—'

        rows += f"""
        <tr style="border-bottom:1px solid #21262d;{row_border}">
          <td style="padding:10px 12px;font-weight:bold;font-size:15px">{t['ticker']}</td>
          <td style="padding:10px 12px;color:#8b949e;font-size:13px">{t['entry_date']}</td>
          <td style="padding:10px 12px;font-size:13px">{rsi_cells}</td>
          <td style="padding:10px 12px">${t['entry_stock']:,.2f}</td>
          <td style="padding:10px 12px">${t['strike']:.0f} C</td>
          <td style="padding:10px 12px">{t['sigma']*100:.0f}%</td>
          <td style="padding:10px 12px">{rank_cell}</td>
          <td style="padding:10px 12px">${t['entry_price']:.2f} → {ep_str}{est_note}</td>
          <td style="padding:10px 12px;color:#8b949e">{hold}d</td>
          <td style="padding:10px 12px">{pnl_str}</td>
          <td style="padding:10px 12px">{badge}</td>
        </tr>"""

    def card(label, value, sub='', color='#c9d1d9'):
        return (
            f'<div style="flex:1;min-width:140px;background:#161b22;border:1px solid #30363d;'
            f'border-radius:8px;padding:16px 18px">'
            f'<div style="color:#8b949e;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:0.8px;margin-bottom:8px">{label}</div>'
            f'<div style="font-size:24px;font-weight:bold;color:{color}">{value}</div>'
            f'<div style="color:#8b949e;font-size:11px;margin-top:4px">{sub}</div></div>'
        )

    best_str  = f'{best_pnl:+.0f}%'  if best_pnl  is not None else '—'
    worst_str = f'{worst_pnl:+.0f}%' if worst_pnl is not None else '—'

    filter_sub = (f"filtered {len(filtered_out)} high-IV signals"
                  if iv_rank_max is not None else "no IV filter applied")
    filter_card = (
        card("IV Filtered", len(filtered_out),
             f"IV Rank ≥ {iv_rank_max} rejected", '#d29922')
        if iv_rank_max is not None
        else card("IV Filter", "off", "run with --iv-rank-max 40", '#8b949e')
    )

    summary = f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:26px">
      {card("Signals entered", len(trades), f"{years}y window · {filter_sub}")}
      {card("Win Rate", f"{win_rate:.0f}%", f"{len(wins)}W / {len(losses)}L / {len(opens)} open",
            '#3fb950' if win_rate >= 50 else '#f85149')}
      {card("Avg P&L (closed)", f"{avg_pnl_cl:+.1f}%", "+50% / −30% exit",
            '#3fb950' if avg_pnl_cl >= 0 else '#f85149')}
      {card("Avg P&L (all)", f"{avg_pnl_all:+.1f}%", "incl. open (est.)",
            '#3fb950' if avg_pnl_all >= 0 else '#f85149')}
      {card("Best / Worst", f"{best_str} / {worst_str}", "closed trades",
            '#3fb950' if (best_pnl or 0) > 0 else '#c9d1d9')}
      {card("Avg Hold", f"{avg_hold}d", "closed trades only", '#58a6ff')}
      {filter_card}
    </div>"""

    stop_str      = (f"stock stop −{stock_stop*100:.0f}%" if stock_stop
                     else f"option stop −{int(LOSS_TARGET*-100)}%")
    iv_filter_str = f"IV Rank &lt; {iv_rank_max}" if iv_rank_max is not None else "no IV filter"
    filter_banner = ""
    if iv_rank_max is not None and filtered_out:
        rows_filtered = "".join(
            f'<tr style="border-bottom:1px solid #21262d;opacity:0.55">'
            f'<td style="padding:8px 12px;font-weight:bold">{f["ticker"]}</td>'
            f'<td style="padding:8px 12px;color:#8b949e">{f["date"]}</td>'
            f'<td style="padding:8px 12px">RSI {f["rsi"]:.1f}</td>'
            f'<td style="padding:8px 12px;color:#f85149">IV Rank {f["iv_rank"]:.0f} ≥ {iv_rank_max} — skipped</td>'
            f'</tr>'
            for f in filtered_out
        )
        filter_banner = f"""
        <div style="margin-bottom:22px">
          <div style="color:#d29922;font-size:12px;text-transform:uppercase;
               letter-spacing:0.8px;margin-bottom:8px">
            Filtered Signals — RSI fired but IV Rank ≥ {iv_rank_max} (too expensive to enter)
          </div>
          <div style="overflow-x:auto">
          <table style="font-size:12px;border-collapse:collapse;width:100%;
                 background:#161b22;border:1px solid #30363d;border-radius:6px;overflow:hidden">
            <thead><tr style="background:#21262d">
              <th style="padding:7px 12px;text-align:left;color:#8b949e;font-weight:normal">Ticker</th>
              <th style="padding:7px 12px;text-align:left;color:#8b949e;font-weight:normal">Signal Date</th>
              <th style="padding:7px 12px;text-align:left;color:#8b949e;font-weight:normal">RSI</th>
              <th style="padding:7px 12px;text-align:left;color:#8b949e;font-weight:normal">Reason Skipped</th>
            </tr></thead>
            <tbody>{rows_filtered}</tbody>
          </table>
          </div>
        </div>"""

    iv_rank_legend = """
    <div style="font-size:11px;color:#8b949e;margin-bottom:18px">
      IV Rank legend: &nbsp;
      <span style="color:#3fb950">■ &lt;20 IDEAL</span> &nbsp;
      <span style="color:#56d364">■ 20–40 GOOD</span> &nbsp;
      <span style="color:#d29922">■ 40–60 FAIR</span> &nbsp;
      <span style="color:#e3a341">■ 60–80 ABOVE AVG</span> &nbsp;
      <span style="color:#f85149">■ &gt;80 EXPENSIVE</span>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RSI Long Backtest {years}y{f" IVR{iv_rank_max}" if iv_rank_max else ""} — {today_str}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0d1117; color:#c9d1d9; font-family:'Courier New',monospace; }}
  .wrap {{ max-width:1200px; margin:0 auto; padding:36px 24px; }}
  h1 {{ font-size:22px; color:#58a6ff; margin-bottom:6px; }}
  .sub {{ color:#8b949e; font-size:13px; margin-bottom:26px; line-height:1.6; }}
  .note {{ background:#161b22; border:1px solid #30363d; border-radius:6px;
           padding:13px 17px; margin-bottom:22px; font-size:12px; color:#8b949e; line-height:1.7; }}
  .tw {{ overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; background:#161b22;
           border:1px solid #30363d; border-radius:8px; overflow:hidden; }}
  thead tr {{ background:#21262d; }}
  th {{ padding:9px 12px; text-align:left; color:#8b949e; font-weight:normal;
        font-size:11px; text-transform:uppercase; letter-spacing:0.6px; white-space:nowrap; }}
  tbody tr:hover {{ background:#1c2128; }}
  .footer {{ color:#8b949e; font-size:11px; margin-top:20px;
             border-top:1px solid #21262d; padding-top:11px; text-align:center; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>RSI Oversold LEAPS Call Backtest — Long ({years:.0f} Years){f" · IV Rank &lt; {iv_rank_max}" if iv_rank_max else ""}</h1>
  <div class="sub">
    {today_str} &nbsp;·&nbsp; {years:.0f}-year window &nbsp;·&nbsp;
    Universe: top 40 large-cap stocks &nbsp;·&nbsp;
    Signal: RSI({RSI_PERIOD}) &lt; {RSI_THRESHOLD:.0f} on any of last {RSI_LOOKBACK} days
    (first occurrence per ticker) &nbsp;·&nbsp;
    LEAPS: ≥{MIN_LEAPS_DAYS}d ITM call Δ≈{target_delta} &nbsp;·&nbsp;
    <span style="color:#58a6ff">{stop_str}</span> &nbsp;·&nbsp;
    <span style="color:{'#3fb950' if iv_rank_max else '#8b949e'}">{iv_filter_str}</span>
  </div>
  {summary}
  <div class="note">
    <span style="color:#79c0ff;font-weight:bold">Methodology &amp; Assumptions</span><br>
    · Black-Scholes call pricing using 30-day realized vol at entry as IV proxy; IV held constant<br>
    · Strike = nearest $5 to Δ≈{target_delta} (ITM, K &lt; S); synthetic ≥{MIN_LEAPS_DAYS}-day expiry<br>
    · IV Rank = where current 30d realized vol sits in its 52-week range (0=cheapest, 100=most expensive)<br>
    · One trade per ticker — first RSI signal in the window only (re-entry not modelled)<br>
    · No bid/ask spread, commissions, or liquidity constraints; RSI uses Wilder's smoothing
  </div>
  {filter_banner}
  {iv_rank_legend}
  <div class="tw">
  <table>
    <thead>
      <tr>
        <th>Ticker</th><th>Entry Date</th><th>RSI (t / t−1 / t−2)</th>
        <th>Stock at Entry</th><th>Strike</th><th>IV (entry)</th>
        <th>IV Rank</th>
        <th>Option Entry → Exit</th><th>Hold</th><th>P&amp;L</th><th>Outcome</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
  <p class="footer">
    RSI Long Backtest &nbsp;·&nbsp; Not financial advice &nbsp;·&nbsp;
    Black-Scholes estimates only, constant IV assumed &nbsp;·&nbsp; Data: Yahoo Finance
  </p>
</div>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    return output_path


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RSI Oversold LEAPS Call Backtest (Long)')
    parser.add_argument('--years',       type=float, default=2.0,
                        help='Backtest window in years (default: 2, max: 3)')
    parser.add_argument('--delta',       type=float, default=0.70,
                        help='Target call delta (default: 0.70)')
    parser.add_argument('--iv-rank-max', type=float, default=None,
                        help='Max IV Rank at entry (e.g. 40). Signals with IV Rank ≥ this are skipped.')
    parser.add_argument('--stock-stop',  type=float, default=None,
                        help='Stop loss as fraction of entry STOCK price (e.g. 0.10 = 10%% stock drop). '
                             'Replaces the default -30%% option-level stop. Recommended: 0.10')
    parser.add_argument('--output',      type=str,   default='',
                        help='Output HTML path (auto-named if omitted)')
    args = parser.parse_args()

    years        = min(args.years, 3.0)
    target_delta = args.delta
    iv_rank_max  = args.iv_rank_max
    stock_stop   = args.stock_stop
    ivr_tag      = f'_ivr{int(iv_rank_max)}' if iv_rank_max is not None else ''
    stp_tag      = f'_stp{int(stock_stop*100)}' if stock_stop is not None else ''
    output_path  = args.output or f'rsi_backtest_long_{years:.0f}y{ivr_tag}{stp_tag}.html'

    print(f"RSI Long Backtest | {years:.0f}-year window | Δ≈{target_delta} | "
          f"RSI < {RSI_THRESHOLD} | stop: {'stock −'+str(int(stock_stop*100))+'%' if stock_stop else 'option −30%'} | "
          f"IV filter: {'< '+str(iv_rank_max) if iv_rank_max else 'off'}")

    trades, filtered_out = run_backtest(years, target_delta, iv_rank_max, stock_stop)

    if not trades:
        print("\nNo signals passed the filters.")
    else:
        path = generate_html(trades, filtered_out, years, target_delta,
                             iv_rank_max, stock_stop, output_path)
        print(f"\nSaved → {path}")
        print(f"Open in browser: open {path}")
