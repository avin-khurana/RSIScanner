#!/usr/bin/env python3
"""
RSI Oversold + LEAPS IV Scanner
Fetches top 40 large-cap stocks (market cap > $100B), filters for RSI(14) < 30,
then checks LEAPS IV for ITM calls with >= 360 days to expiry.
Sends a rich HTML email with IV Rank, raw IV, vs-average, and vega-gain estimate.
"""

import os
import smtplib
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MIN_MARKET_CAP = 100e9   # $100B
TOP_N          = 40
RSI_PERIOD     = 14
RSI_THRESHOLD  = 30.0
RSI_LOOKBACK   = 3      # signal if RSI < threshold on any of the last N trading days
TARGET_DELTA   = 0.70    # ITM call delta target for LEAPS
MIN_LEAPS_DAYS = 360
RISK_FREE_RATE = 0.045

EMAIL_FROM     = os.environ.get('EMAIL_FROM', '')
EMAIL_PASSWORD = os.environ.get('EMAIL_APP_PASSWORD', '')
EMAIL_TO       = os.environ.get('EMAIL_TO', 'avin.khurana18@gmail.com')

# Fallback list used if Wikipedia fetch fails
_LARGE_CAP_FALLBACK = [
    'AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','BRK-B',
    'UNH','XOM','JPM','JNJ','V','PG','MA','HD','CVX','MRK','ABBV',
    'LLY','COST','PEP','KO','AVGO','BAC','WMT','MCD','CRM','NFLX',
    'TMO','AMD','ACN','NEE','ORCL','QCOM','TXN','IBM','GS','SPGI','LIN',
    'CAT','BLK','AXP','GE','AMGN','HON','RTX','PM','DHR','SCHW',
]


# ─── UNIVERSE ────────────────────────────────────────────────────────────────
def get_large_cap_universe():
    """Returns top TOP_N tickers with market cap > MIN_MARKET_CAP, fetched dynamically."""
    try:
        tables = pd.read_html(
            'https://en.wikipedia.org/wiki/S%26P_100',
            attrs={'id': 'constituents'}
        )
        sp100 = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
    except Exception:
        sp100 = []

    candidates = list(dict.fromkeys(sp100 + _LARGE_CAP_FALLBACK))[:120]
    print(f"  Fetching market caps for {len(candidates)} candidates...")

    def fetch_mcap(ticker):
        try:
            fi = yf.Ticker(ticker).fast_info
            mc = getattr(fi, 'market_cap', None) or 0
            return ticker, float(mc)
        except Exception:
            return ticker, 0.0

    with ThreadPoolExecutor(max_workers=20) as ex:
        pairs = list(ex.map(fetch_mcap, candidates))

    pairs = [(t, mc) for t, mc in pairs if mc > MIN_MARKET_CAP]
    pairs.sort(key=lambda x: -x[1])
    tickers  = [t for t, _ in pairs[:TOP_N]]
    mcap_map = {t: mc for t, mc in pairs[:TOP_N]}

    print(f"  Universe: {len(tickers)} tickers with MCap > ${MIN_MARKET_CAP/1e9:.0f}B")
    print(f"  {', '.join(tickers)}\n")
    return tickers, mcap_map


# ─── INDICATORS ──────────────────────────────────────────────────────────────
def fix_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def calc_rsi(closes, period=RSI_PERIOD):
    # Wilder's smoothing (EMA with com=period-1) — matches TradingView's RSI exactly
    delta    = closes.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))

def realized_vol(closes, window=30):
    return np.log(closes / closes.shift(1)).dropna().rolling(window).std() * np.sqrt(252)

def iv_stats(rv_series, lookback=252):
    """Returns (iv_rank, iv_lo_pct, iv_hi_pct, iv_avg_pct) from realized vol series."""
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


# ─── BLACK-SCHOLES ────────────────────────────────────────────────────────────
def bs_greeks(S, K, T, r=RISK_FREE_RATE, sigma=0.30):
    if T <= 1e-6 or S <= 0 or K <= 0 or sigma <= 0:
        return None
    d1  = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2  = d1 - sigma * np.sqrt(T)
    pdf = norm.pdf(d1)
    return dict(
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
        delta = norm.cdf(d1),
        gamma = pdf / (S * sigma * np.sqrt(T)),
        theta = (-(S * pdf * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365,
        vega  = S * pdf * np.sqrt(T) / 100,
    )

def find_strike_for_delta(S, target_delta, T, r, sigma):
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

def iv_rank_tier(rank):
    """Returns (label, hex_color, guidance) for an IV Rank value."""
    if rank < 20:
        return "IDEAL BUY",   "#3fb950", "IV near 52-wk low — cheapest options of the year"
    if rank < 40:
        return "GOOD",        "#56d364", "IV below average — favorable for buying"
    if rank < 60:
        return "FAIR",        "#d29922", "IV near average — neutral entry"
    if rank < 80:
        return "ABOVE AVG",   "#e3a341", "IV above average — consider spreads over naked longs"
    return     "EXPENSIVE",   "#f85149", "IV near 52-wk high — poor time to buy naked options"

def implied_vol(S, K, T, r, market_price, max_iter=100, tol=1e-6):
    """Newton-Raphson IV solver: finds sigma such that BS_call(sigma) == market_price."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if market_price < intrinsic - 0.01:
        return None
    sigma = 0.30
    for _ in range(max_iter):
        g = bs_greeks(S, K, T, r, sigma)
        if g is None:
            return None
        diff = g['price'] - market_price
        if abs(diff) < tol:
            return sigma
        raw_vega = g['vega'] * 100  # bs_greeks stores vega per 1% IV; convert to per-unit-sigma
        if raw_vega < 1e-8:
            break
        sigma -= diff / raw_vega
        sigma = max(1e-4, min(sigma, 20.0))
    g = bs_greeks(S, K, T, r, sigma)
    return sigma if (g and abs(g['price'] - market_price) < 0.10) else None


# ─── LEAPS IV ────────────────────────────────────────────────────────────────
def get_leaps_itm(ticker_obj, price, curr_iv):
    """
    Finds the nearest expiry with >= MIN_LEAPS_DAYS remaining.
    Returns the ITM call closest to TARGET_DELTA (0.70), plus expiry date and T (years).
    Falls back to Black-Scholes if the chain is unavailable.
    """
    try:
        exps = ticker_obj.options
        if not exps:
            return None, None, None

        today      = datetime.now()
        valid_exps = [
            e for e in exps
            if (datetime.strptime(e, '%Y-%m-%d') - today).days >= MIN_LEAPS_DAYS
        ]
        if not valid_exps:
            return None, None, None

        # Nearest qualifying expiry
        best_exp = min(valid_exps, key=lambda e: (datetime.strptime(e, '%Y-%m-%d') - today).days)
        T = (datetime.strptime(best_exp, '%Y-%m-%d') - today).days / 365

        calls = ticker_obj.option_chain(best_exp).calls.copy()
        # ITM: strike < price; keep rows even when Yahoo's impliedVolatility is NaN/0
        calls = calls[calls['strike'] < price].dropna(subset=['strike'])
        calls = calls[calls['strike'] > 0]
        if calls.empty:
            return None, best_exp, T

        calls['bs_delta'] = calls.apply(
            lambda row: (
                bs_greeks(
                    price, row['strike'], T, RISK_FREE_RATE,
                    row['impliedVolatility']
                    if pd.notna(row.get('impliedVolatility')) and row['impliedVolatility'] > 0
                    else curr_iv,
                ) or {}
            ).get('delta', 0),
            axis=1,
        )
        best = calls.iloc[(calls['bs_delta'] - TARGET_DELTA).abs().argsort()[:1]]
        return best.iloc[0], best_exp, T

    except Exception:
        return None, None, None


# ─── SCAN ONE TICKER ─────────────────────────────────────────────────────────
def scan_ticker(ticker, mcap):
    result = dict(ticker=ticker, mcap=mcap, error=None, rsi_signal=False)
    try:
        df = fix_cols(yf.download(ticker, period='2y', progress=False, auto_adjust=True))
        if len(df) < 60:
            result['error'] = 'insufficient data'
            return result

        closes     = df['Close'].squeeze()
        rsi_series = calc_rsi(closes).dropna()
        price      = float(closes.iloc[-1])
        rv         = realized_vol(closes, 30)
        curr_iv    = float(rv.dropna().iloc[-1]) if not rv.dropna().empty else 0.30

        # Last RSI_LOOKBACK trading-day RSI values: index 0 = today, 1 = yesterday, etc.
        rsi_days = [
            round(float(rsi_series.iloc[-i]), 1) if len(rsi_series) >= i else None
            for i in range(1, RSI_LOOKBACK + 1)
        ]
        rsi_val = rsi_days[0]   # today's RSI, used for all downstream calculations

        iv_rank, iv_lo, iv_hi, iv_avg = iv_stats(rv)

        result.update(dict(
            price    = round(price, 2),
            rsi      = rsi_val,
            rsi_days = rsi_days,           # list: [today, yesterday, 2-days-ago]
            curr_iv  = round(curr_iv * 100, 1),
            iv_rank  = iv_rank,
            iv_lo    = iv_lo,
            iv_hi    = iv_hi,
            iv_avg   = iv_avg,
        ))
        # Signal fires if RSI was below threshold on ANY of the last RSI_LOOKBACK days
        result['rsi_signal'] = any(v is not None and v < RSI_THRESHOLD for v in rsi_days)

        if result['rsi_signal']:
            t_obj = yf.Ticker(ticker)
            chain_row, exp_date, T = get_leaps_itm(t_obj, price, curr_iv)

            if chain_row is not None:
                strike    = float(chain_row['strike'])
                bid       = float(chain_row.get('bid') or 0)
                ask       = float(chain_row.get('ask') or 0)
                mid       = (bid + ask) / 2
                last      = float(chain_row.get('lastPrice') or 0)
                prem      = mid if mid > 0 else last

                # Prefer IV back-solved from market price; fall back to Yahoo's field, then realized vol
                yahoo_iv  = float(chain_row.get('impliedVolatility') or 0)
                solved_iv = implied_vol(price, strike, T, RISK_FREE_RATE, prem) if prem > 0 else None
                chain_iv  = solved_iv or (yahoo_iv if yahoo_iv > 0 else curr_iv)

                greeks = bs_greeks(price, strike, T, RISK_FREE_RATE, chain_iv)
                if prem <= 0 and greeks:
                    prem = greeks['price']
                if prem <= 0:
                    prem = curr_iv * 0.40 * price
            else:
                T        = T or 1.0
                chain_iv = curr_iv
                exp_date = exp_date or f'+{MIN_LEAPS_DAYS}d (est.)'
                strike   = find_strike_for_delta(price, TARGET_DELTA, T, RISK_FREE_RATE, curr_iv)
                greeks   = bs_greeks(price, strike, T, RISK_FREE_RATE, curr_iv)
                prem     = greeks['price'] if greeks else curr_iv * 0.40 * price

            prem = prem if prem > 0 else curr_iv * 0.40 * price

            # Vega gain: estimated option gain if IV reverts from current to 52-week average
            iv_expand_gain = 0.0
            if greeks and greeks.get('vega') and prem > 0:
                iv_delta_pts   = max(0.0, iv_avg - chain_iv * 100)
                iv_expand_gain = round(greeks['vega'] * iv_delta_pts / prem * 100, 1)

            result.update(dict(
                leaps_exp      = exp_date,
                leaps_days     = round((T or 0) * 365),
                leaps_strike   = round(strike, 0),
                leaps_iv       = round(chain_iv * 100, 1),
                leaps_prem     = round(prem, 2),
                leaps_delta    = round(greeks['delta'], 2) if greeks else None,
                leaps_theta    = round(greeks['theta'], 2) if greeks else None,
                leaps_vega     = round(greeks['vega'],  2) if greeks else None,
                leaps_be       = round(strike + prem, 2),
                leaps_be_pct   = round((strike + prem) / price * 100 - 100, 1),
                iv_expand_gain = iv_expand_gain,
            ))

    except Exception as e:
        result['error'] = str(e)

    return result


# ─── EMAIL ───────────────────────────────────────────────────────────────────
def send_email(subject, html_body):
    if not EMAIL_FROM or not EMAIL_PASSWORD:
        print("  Email skipped — set EMAIL_FROM and EMAIL_APP_PASSWORD env vars.")
        return
    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(EMAIL_FROM, EMAIL_PASSWORD)
            s.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print(f"  Email sent to {EMAIL_TO}.")
    except Exception as e:
        print(f"  Email error: {e}")


def build_email_html(signals, all_valid, mcap_map, run_date):
    BASE  = "background:#0d1117;color:#c9d1d9;font-family:'Courier New',monospace;padding:0;margin:0"
    WRAP  = "max-width:720px;margin:0 auto;padding:24px"

    header = (
        f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
        f'padding:16px 20px;margin-bottom:24px">'
        f'<div style="font-size:18px;font-weight:bold;color:#58a6ff;margin-bottom:6px">'
        f'RSI Oversold Scanner</div>'
        f'<div style="color:#8b949e;font-size:13px">{run_date} &nbsp;·&nbsp; '
        f'Top {TOP_N} stocks · MCap &gt; ${MIN_MARKET_CAP/1e9:.0f}B · '
        f'Scanned {len(all_valid)} · '
        f'<span style="color:{"#f85149" if signals else "#8b949e"}">'
        f'RSI &lt; {RSI_THRESHOLD:.0f} (any of last {RSI_LOOKBACK}d): <b>{len(signals)}</b></span></div>'
        f'</div>'
    )

    # ── Signal cards ──
    if not signals:
        cards_html = (
            '<div style="text-align:center;color:#8b949e;padding:40px;'
            'background:#161b22;border:1px solid #30363d;border-radius:8px">'
            f'No stocks with RSI &lt; {RSI_THRESHOLD:.0f} today.</div>'
        )
    else:
        cards_html = ""
        for r in sorted(signals, key=lambda x: x['rsi']):
            mcap_b                        = mcap_map.get(r['ticker'], 0) / 1e9
            tier_label, tier_color, tier_guidance = iv_rank_tier(r['iv_rank'])
            iv_rank_color                 = tier_color

            if r.get('leaps_strike'):
                iv_rank_label = (
                    f'<b style="color:{tier_color}">{r["iv_rank"]:.0f}/100 — {tier_label}</b>'
                    f'<br><span style="color:#8b949e;font-size:11px">{tier_guidance}</span>'
                    f'<br><span style="color:#8b949e;font-size:11px">'
                    f'Ideal: &lt;20 &nbsp;·&nbsp; Good: 20–40 &nbsp;·&nbsp; '
                    f'Fair: 40–60 &nbsp;·&nbsp; Above avg: 60–80 &nbsp;·&nbsp; Expensive: &gt;80</span>'
                )
                iv_expand_str = (
                    f' &nbsp;·&nbsp; <b style="color:#3fb950">+{r["iv_expand_gain"]:.0f}% option gain '
                    f'if IV reverts to avg</b>'
                    if r.get('iv_expand_gain', 0) > 0 else ''
                )
                gk_str = (
                    f'Δ{r["leaps_delta"]} &nbsp; Θ${r["leaps_theta"]}/day &nbsp; V${r["leaps_vega"]}/1%&nbsp;IV'
                    if r.get('leaps_delta') else 'N/A'
                )
                leaps_html = f"""
<div style="background:#0d1a0d;border-left:3px solid #238636;border-radius:4px;
     padding:14px;margin-top:12px">
  <div style="color:#3fb950;font-weight:bold;font-size:13px;margin-bottom:10px">
    LEAPS — {r['leaps_days']}-Day ITM Call &nbsp;
    <span style="color:#8b949e;font-weight:normal">(≥{MIN_LEAPS_DAYS}d · Δ≈{TARGET_DELTA})</span>
  </div>
  <table style="font-size:13px;border-collapse:collapse;width:100%">
    <tr>
      <td style="color:#8b949e;width:140px;padding:4px 0;vertical-align:top">Strike / Expiry</td>
      <td style="padding:4px 0"><b>${r['leaps_strike']:.0f} Call</b> &nbsp;({r['leaps_exp']})</td>
    </tr>
    <tr>
      <td style="color:#8b949e;padding:4px 0;vertical-align:top">IV (chain)</td>
      <td style="padding:4px 0">
        <b style="color:{iv_rank_color}">{r['leaps_iv']:.0f}%</b>
        &nbsp;·&nbsp; Rank {iv_rank_label}
        <br>
        <span style="color:#8b949e;font-size:12px">52-wk: </span>
        <span style="color:#3fb950">{r['iv_lo']:.0f}%</span>
        <span style="color:#8b949e"> → avg </span>
        <span style="color:#d29922">{r['iv_avg']:.0f}%</span>
        <span style="color:#8b949e"> → </span>
        <span style="color:#f85149">{r['iv_hi']:.0f}%</span>
        {iv_expand_str}
      </td>
    </tr>
    <tr>
      <td style="color:#8b949e;padding:4px 0">Premium</td>
      <td style="padding:4px 0">
        <b>${r['leaps_prem']:.2f}/share</b>
        &nbsp;(${r['leaps_prem']*100:,.0f}/contract)
      </td>
    </tr>
    <tr>
      <td style="color:#8b949e;padding:4px 0">Greeks</td>
      <td style="padding:4px 0">{gk_str}</td>
    </tr>
    <tr>
      <td style="color:#8b949e;padding:4px 0">Breakeven</td>
      <td style="padding:4px 0">
        ${r['leaps_be']:,.2f}
        &nbsp;<span style="color:#d29922">({r['leaps_be_pct']:+.1f}% above today)</span>
      </td>
    </tr>
  </table>
</div>"""
            else:
                leaps_html = (
                    '<div style="color:#8b949e;font-size:12px;margin-top:10px;'
                    'padding:10px;background:#161b22;border-radius:4px">'
                    f'No LEAPS chain with ≥{MIN_LEAPS_DAYS} days found for this ticker.</div>'
                )

            cards_html += f"""
<div style="background:#161b22;border:1px solid #30363d;border-left:4px solid #f85149;
     border-radius:8px;padding:20px;margin-bottom:18px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
    <div>
      <span style="font-size:22px;font-weight:bold">{r['ticker']}</span>
      &nbsp;<span style="color:#8b949e;font-size:13px">MCap ${mcap_b:.0f}B · ${r['price']:,.2f}</span>
    </div>
    <span style="background:#f85149;color:#000;padding:5px 14px;border-radius:12px;
          font-size:13px;font-weight:bold">RSI {r['rsi']:.1f}</span>
  </div>
  <div style="display:flex;gap:24px;font-size:13px;margin-bottom:10px;flex-wrap:wrap">
    <div>
      <span style="color:#8b949e">RSI(14) — last {RSI_LOOKBACK} days: </span>
      {' &nbsp;'.join(
        f'<b style="color:{"#f85149" if v is not None and v < RSI_THRESHOLD else "#d29922" if v is not None and v < 35 else "#c9d1d9"}">'
        f'{"Today" if i == 0 else ("-1d" if i == 1 else "-2d")}:{v:.1f}{"★" if v is not None and v < RSI_THRESHOLD else ""}</b>'
        for i, v in enumerate(r['rsi_days'])
        if v is not None
      )}
      <span style="color:#8b949e;font-size:11px"> (★ = below {RSI_THRESHOLD:.0f})</span>
    </div>
  </div>
  <div style="display:flex;gap:32px;font-size:13px;margin-bottom:6px">
    <div><span style="color:#8b949e">Realized IV: </span><b>{r['curr_iv']:.0f}%</b></div>
    <div><span style="color:#8b949e">IV Rank: </span>
         <b style="color:{iv_rank_color}">{r['iv_rank']:.0f}/100</b></div>
  </div>
  {leaps_html}
</div>"""

    # ── Near misses table (RSI 30–35) ──
    # Near misses: not a signal yet, but at least one day in last 3 had RSI 30–35
    def _near_miss(r):
        if r.get('rsi_signal'):
            return False
        return any(v is not None and 30 <= v <= 35 for v in r.get('rsi_days', []))

    near = sorted(
        [r for r in all_valid if _near_miss(r)],
        key=lambda x: min((v for v in x['rsi_days'] if v is not None), default=99),
    )
    near_html = ""
    if near:
        def _rsi_cell(v):
            if v is None:
                return '<span style="color:#8b949e">—</span>'
            color = "#f85149" if v < RSI_THRESHOLD else ("#d29922" if v < 35 else "#c9d1d9")
            return f'<span style="color:{color}">{v:.1f}</span>'

        def _rank_cell(rank):
            lbl, col, _ = iv_rank_tier(rank)
            return f'<span style="color:{col}">{rank:.0f}/100 {lbl}</span>'

        rows = "".join(
            f'<tr>'
            f'<td style="padding:6px 10px;font-weight:bold">{r["ticker"]}</td>'
            f'<td style="padding:6px 10px">'
            + " / ".join(_rsi_cell(v) for v in r.get("rsi_days", [r["rsi"], None, None]))
            + f'</td>'
            f'<td style="padding:6px 10px">${r["price"]:,.2f}</td>'
            f'<td style="padding:6px 10px">{_rank_cell(r["iv_rank"])}</td>'
            f'<td style="padding:6px 10px">${mcap_map.get(r["ticker"],0)/1e9:.0f}B</td>'
            f'</tr>'
            for r in near
        )
        near_html = f"""
<div style="margin-top:28px">
  <div style="color:#79c0ff;font-size:12px;text-transform:uppercase;letter-spacing:1px;
       margin-bottom:10px">Near Misses — RSI 30–35 in last {RSI_LOOKBACK} days (Watch List)</div>
  <div style="overflow-x:auto">
  <table style="font-size:13px;border-collapse:collapse;width:100%;
         background:#161b22;border:1px solid #30363d;border-radius:6px">
    <thead>
      <tr style="background:#21262d">
        <th style="padding:8px 10px;text-align:left;color:#8b949e;font-weight:normal">Ticker</th>
        <th style="padding:8px 10px;text-align:left;color:#8b949e;font-weight:normal">RSI (today / -1d / -2d)</th>
        <th style="padding:8px 10px;text-align:left;color:#8b949e;font-weight:normal">Price</th>
        <th style="padding:8px 10px;text-align:left;color:#8b949e;font-weight:normal">IV Rank</th>
        <th style="padding:8px 10px;text-align:left;color:#8b949e;font-weight:normal">MCap</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
</div>"""

    footer = (
        '<p style="color:#8b949e;font-size:11px;margin-top:28px;'
        'border-top:1px solid #21262d;padding-top:12px;text-align:center">'
        'RSI Oversold Scanner · Not financial advice · Data: Yahoo Finance</p>'
    )

    return f'<html><body style="{BASE}"><div style="{WRAP}">{header}{cards_html}{near_html}{footer}</div></body></html>'


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    run_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    bar = '═' * 68
    print(f"\n{bar}")
    print(f"  RSI Oversold Scanner — {run_date}")
    print(f"  Filter : RSI({RSI_PERIOD}) < {RSI_THRESHOLD}")
    print(f"  LEAPS  : ≥{MIN_LEAPS_DAYS}-day ITM calls · Δ ≈ {TARGET_DELTA}")
    print(f"{bar}\n")

    tickers, mcap_map = get_large_cap_universe()

    print(f"  Scanning {len(tickers)} tickers...\n")
    results = []
    for ticker in tickers:
        print(f"    {ticker:<6} ...", end='', flush=True)
        r = scan_ticker(ticker, mcap_map.get(ticker, 0))
        results.append(r)
        if r.get('error'):
            print(f" ERR: {r['error']}")
        elif r['rsi_signal']:
            rsi_str = ' / '.join(f'{v:.1f}' if v is not None else '—' for v in r['rsi_days'])
            print(f" RSI=[{rsi_str}]  <-- OVERSOLD (last {RSI_LOOKBACK}d)")
        else:
            print(f" RSI={r['rsi']:.1f}")

    signals = [r for r in results if r.get('rsi_signal') and not r.get('error')]
    valid   = [r for r in results if not r.get('error')]

    print(f"\n{bar}")
    print(f"  RESULTS: {len(signals)} signal(s) with RSI < {RSI_THRESHOLD} out of {len(valid)} scanned")
    print(bar)
    for r in sorted(signals, key=lambda x: x['rsi']):
        rsi_str   = ' / '.join(f'{v:.1f}' if v is not None else '—' for v in r['rsi_days'])
        tier, _, guidance = iv_rank_tier(r['iv_rank'])
        print(f"  {r['ticker']:<6}  RSI(last {RSI_LOOKBACK}d)=[{rsi_str}]  "
              f"Price=${r['price']:,.2f}  IV Rank={r['iv_rank']:.0f}/100 [{tier}]")
        print(f"           IV guidance: {guidance}")
        print(f"           IV scale: <20 IDEAL BUY · 20-40 GOOD · 40-60 FAIR · 60-80 ABOVE AVG · >80 EXPENSIVE")
        if r.get('leaps_strike'):
            print(f"           LEAPS: ${r['leaps_strike']:.0f} Call ({r['leaps_exp']})  "
                  f"IV={r['leaps_iv']:.0f}%  Rank={r['iv_rank']:.0f} [{tier}]  "
                  f"Prem=${r['leaps_prem']:.2f}  BE={r['leaps_be_pct']:+.1f}%")

    # Save CSV
    csv_path = 'rsi_scan_results.csv'
    pd.DataFrame([{k: v for k, v in r.items()} for r in valid]).to_csv(csv_path, index=False)
    print(f"\n  Saved → {csv_path}")

    # Send email
    html    = build_email_html(signals, valid, mcap_map, run_date)
    tickers_str = ', '.join(r['ticker'] for r in sorted(signals, key=lambda x: x['rsi']))
    subject = (
        f"RSI Oversold {run_date} — {len(signals)} stock(s): {tickers_str}"
        if signals else
        f"RSI Scanner {run_date} — No RSI < {RSI_THRESHOLD:.0f} today"
    )
    send_email(subject, html)
    print(f"\n{bar}\n")


if __name__ == '__main__':
    main()
