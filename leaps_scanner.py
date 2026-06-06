#!/usr/bin/env python3
"""
LEAPS Unified Scanner
Combines the RSI-oversold contrarian strategy (rsi_scanner.py) with the
multi-factor momentum strategy (v3.0) into one daily scan.

Two signal sections per run:
  OVERSOLD  — RSI(14) < 30 on any of last 3 days. No IV filter (proven to hurt
               win rate on contrarian plays). Stock target/stop via BS back-solve.
  MOMENTUM  — RSI 38–62 + SMA200 + MACD cross + volume spike + fundamentals.
               IV Rank < 40 and VIX < 28 gates applied (appropriate for
               trend-following entries where option cost matters more).

Both sections surface:
  • 3-month ATM call  (momentum / quick-profit play)
  • 12-month LEAPS    (Δ≈0.70, trend play with more time buffer)
  • Stock target ▲ and stock stop ▼ for trade management
  • Ready-to-paste row for my_trades.csv / trade_tracker.py

Usage:
  python leaps_scanner.py               # full scan + email + telegram
  python leaps_scanner.py --no-email    # scan only, print to stdout
"""

import os
import smtplib
import resource
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Universe (dynamic, same as rsi_scanner.py)
MIN_MARKET_CAP  = 100e9
TOP_N           = 40

# RSI / Oversold
RSI_PERIOD      = 14
RSI_THRESHOLD   = 30.0      # oversold signal threshold
RSI_LOOKBACK    = 3         # fires if RSI < threshold on any of last N days

# LEAPS
TARGET_DELTA    = 0.70
MIN_LEAPS_DAYS  = 360
RISK_FREE_RATE  = 0.045

# Trade management
PROFIT_TARGET_PCT = 0.50    # +50% option gain → target
STOP_STOCK_PCT    = 0.10    # −10% stock drop → stop

# Momentum filters (applied to MOMENTUM section only, not oversold)
VIX_MAX         = 28
IV_RANK_MAX     = 40
MIN_DAYS_EARN   = 30
RSI_MOM_LO      = 38        # RSI range for momentum signal
RSI_MOM_HI      = 62

# Position sizing
ACCOUNT_SIZE    = 100_000
MAX_POS_PCT     = 0.05

# Alerts
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN',   '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
EMAIL_FROM       = os.environ.get('EMAIL_FROM',        '')
EMAIL_PASSWORD   = os.environ.get('EMAIL_APP_PASSWORD', '')
EMAIL_TO         = os.environ.get('EMAIL_TO',          'avin.khurana18@gmail.com')

_LARGE_CAP_FALLBACK = [
    'AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','BRK-B',
    'UNH','XOM','JPM','JNJ','V','PG','MA','HD','CVX','MRK','ABBV',
    'LLY','COST','PEP','KO','AVGO','BAC','WMT','MCD','CRM','NFLX',
    'TMO','AMD','ACN','NEE','ORCL','QCOM','TXN','IBM','GS','SPGI',
    'LIN','CAT','BLK','AXP','GE','AMGN','HON','RTX','PM','DHR','SCHW',
]


# ── UNIVERSE ──────────────────────────────────────────────────────────────────
def get_large_cap_universe():
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

    with ThreadPoolExecutor(max_workers=8) as ex:
        pairs = list(ex.map(fetch_mcap, candidates))

    pairs = [(t, mc) for t, mc in pairs if mc > MIN_MARKET_CAP]
    pairs.sort(key=lambda x: -x[1])
    tickers  = [t for t, _ in pairs[:TOP_N]]
    mcap_map = {t: mc for t, mc in pairs[:TOP_N]}
    print(f"  Universe: {len(tickers)} tickers  |  {', '.join(tickers)}\n")
    return tickers, mcap_map


# ── INDICATORS ────────────────────────────────────────────────────────────────
def fix_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def calc_rsi(closes, period=RSI_PERIOD):
    """Wilder's smoothing — matches TradingView exactly."""
    delta    = closes.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))

def realized_vol(closes, window=30):
    return np.log(closes / closes.shift(1)).dropna().rolling(window).std() * np.sqrt(252)

def iv_stats(rv_series, lookback=252):
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

def iv_rank_tier(rank):
    if rank < 20: return 'IDEAL BUY',  '#3fb950', 'IV near 52-wk low — cheapest options of the year'
    if rank < 40: return 'GOOD',       '#56d364', 'IV below average — favorable for buying'
    if rank < 60: return 'FAIR',       '#d29922', 'IV near average — neutral entry'
    if rank < 80: return 'ABOVE AVG',  '#e3a341', 'IV above average — consider spreads over naked longs'
    return         'EXPENSIVE',        '#f85149', 'IV near 52-wk high — poor time to buy naked options'

def calc_indicators(df):
    c = df['Close'].squeeze()
    v = df['Volume'].squeeze()
    h = df['High'].squeeze()
    l = df['Low'].squeeze()
    delta    = c.diff()
    avg_gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rsi      = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    macd_line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_sig
    tr    = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return dict(
        c=c, v=v,
        rsi=rsi,
        macd_hist=macd_hist,
        sma50=c.rolling(50).mean(),
        sma200=c.rolling(200).mean(),
        vol50=v.rolling(50).mean(),
        atr=tr.rolling(14).mean(),
        rv=realized_vol(c, 30),
    )


# ── FUNDAMENTALS ──────────────────────────────────────────────────────────────
def score_fundamentals(info):
    score, details = 0, []
    pe = info.get('forwardPE') or info.get('trailingPE')
    if pe and 0 < pe < 20:   score += 20; details.append(f'P/E={pe:.1f}')
    elif pe and pe < 35:      score += 10; details.append(f'P/E={pe:.1f}')
    rg = info.get('revenueGrowth') or 0
    if rg > 0.20:             score += 20; details.append(f'Rev={rg:.0%}')
    elif rg > 0.08:           score += 10; details.append(f'Rev={rg:.0%}')
    mg = info.get('profitMargins') or 0
    if mg > 0.20:             score += 20; details.append(f'Mgn={mg:.0%}')
    elif mg > 0.08:           score += 10
    de = info.get('debtToEquity') or 999
    if 0 < de < 50:           score += 20; details.append('D/E=low')
    elif de < 150:            score += 10
    eg = info.get('earningsGrowth') or 0
    if eg > 0.15:             score += 20; details.append(f'EPS={eg:.0%}')
    elif eg > 0.05:           score += 10
    grade = 'A' if score >= 70 else ('B' if score >= 40 else 'C')
    return grade, score, details

def relative_strength(ticker_df, spy_df, period=63):
    tc = ticker_df['Close'].squeeze()
    sc = spy_df['Close'].squeeze()
    tr = float(tc.iloc[-1] / tc.iloc[-period] - 1)
    sr = float(sc.iloc[-1] / sc.iloc[-period] - 1)
    return tr - sr

def days_to_earnings(t_obj):
    try:
        cal = t_obj.calendar
        if cal is None: return 999
        if isinstance(cal, dict):
            dates = cal.get('Earnings Date', [])
            if dates:
                return max(0, (pd.to_datetime(dates[0]) - pd.Timestamp.now()).days)
        elif isinstance(cal, pd.DataFrame):
            if 'Earnings Date' in cal.index:
                return max(0, (pd.to_datetime(cal.loc['Earnings Date'].iloc[0]) - pd.Timestamp.now()).days)
    except Exception:
        pass
    return 999


# ── BLACK-SCHOLES ─────────────────────────────────────────────────────────────
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
    lo, hi, mid = S * 0.4, S * 1.8, S
    for _ in range(60):
        mid = (lo + hi) / 2
        g = bs_greeks(S, mid, T, r, sigma)
        if g is None: break
        if g['delta'] > target_delta: lo = mid
        else:                         hi = mid
    return round(mid / 5) * 5

def implied_vol(S, K, T, r, market_price, max_iter=100, tol=1e-6):
    """Newton-Raphson IV solver — more accurate than Yahoo's pre-computed field."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if market_price < intrinsic - 0.01:
        return None
    sigma = 0.30
    for _ in range(max_iter):
        g = bs_greeks(S, K, T, r, sigma)
        if g is None: return None
        diff = g['price'] - market_price
        if abs(diff) < tol: return sigma
        raw_vega = g['vega'] * 100
        if raw_vega < 1e-8: break
        sigma -= diff / raw_vega
        sigma = max(1e-4, min(sigma, 20.0))
    g = bs_greeks(S, K, T, r, sigma)
    return sigma if (g and abs(g['price'] - market_price) < 0.10) else None

def stock_target_for_option_gain(S, K, T, r, sigma, entry_price, gain_pct=PROFIT_TARGET_PCT):
    """Binary search: stock price where call gains gain_pct from entry_price."""
    if T <= 0 or sigma <= 0 or entry_price <= 0:
        return round(S * (1 + gain_pct / max(TARGET_DELTA, 0.5)), 2)
    target_opt = entry_price * (1 + gain_pct)
    lo, hi = S, S * 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        g = bs_greeks(mid, K, T, r, sigma)
        if g is None: break
        if g['price'] < target_opt: lo = mid
        else:                       hi = mid
    return round(mid, 2)

def size_position(prem):
    cost = prem * 100
    if cost <= 0: return 0, 0
    n = max(1, int(ACCOUNT_SIZE * MAX_POS_PCT / cost))
    return n, round(n * cost, 0)


# ── OPTIONS ───────────────────────────────────────────────────────────────────
def get_leaps_itm(t_obj, price, curr_iv):
    """Finds ≥MIN_LEAPS_DAYS expiry, ITM call closest to TARGET_DELTA.
    Uses Newton-Raphson IV back-solve for accuracy on deep ITM LEAPS."""
    try:
        exps = t_obj.options
        if not exps: return None, None, None
        today = datetime.now()
        valid = [e for e in exps
                 if (datetime.strptime(e, '%Y-%m-%d') - today).days >= MIN_LEAPS_DAYS]
        if not valid: return None, None, None
        best_exp = min(valid, key=lambda e: (datetime.strptime(e, '%Y-%m-%d') - today).days)
        T = (datetime.strptime(best_exp, '%Y-%m-%d') - today).days / 365
        calls = t_obj.option_chain(best_exp).calls.copy()
        calls = calls[calls['strike'] < price].dropna(subset=['strike'])
        calls = calls[calls['strike'] > 0]
        if calls.empty: return None, best_exp, T
        calls['bs_delta'] = calls.apply(
            lambda row: (bs_greeks(
                price, row['strike'], T, RISK_FREE_RATE,
                row['impliedVolatility']
                if pd.notna(row.get('impliedVolatility')) and row['impliedVolatility'] > 0
                else curr_iv,
            ) or {}).get('delta', 0), axis=1,
        )
        best = calls.iloc[(calls['bs_delta'] - TARGET_DELTA).abs().argsort()[:1]]
        return best.iloc[0], best_exp, T
    except Exception:
        return None, None, None

def get_3m_option(t_obj, price, curr_iv):
    """Nearest expiry to 91 days, ATM call."""
    try:
        exps = t_obj.options
        if not exps: return None, None, None
        target = datetime.now() + timedelta(days=91)
        best_exp = min(exps, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target).days))
        T = (datetime.strptime(best_exp, '%Y-%m-%d') - datetime.now()).days / 365
        if T < 0.05: return None, None, None
        calls = t_obj.option_chain(best_exp).calls.copy()
        calls = calls[calls['strike'].between(price * 0.85, price * 1.15)]
        calls = calls.dropna(subset=['strike', 'impliedVolatility'])
        if calls.empty:
            strike, iv_u, prem = round(price / 5) * 5, curr_iv, 0.0
        else:
            row    = calls.iloc[(calls['strike'] - price).abs().argsort()[:1]].iloc[0]
            strike = float(row['strike'])
            bid    = float(row.get('bid', 0) or 0)
            ask    = float(row.get('ask', 0) or 0)
            mid    = (bid + ask) / 2
            prem   = mid if mid > 0 else float(row.get('lastPrice', 0) or 0)
            iv_u   = float(row['impliedVolatility'])
        g = bs_greeks(price, strike, T, RISK_FREE_RATE, iv_u)
        if prem <= 0 and g: prem = g['price']
        if prem <= 0:       prem = curr_iv * 0.40 * price
        contracts, capital = size_position(prem)
        be_pct = round((strike + prem) / price * 100 - 100, 1)
        return dict(
            strike=round(strike, 0), prem=round(prem, 2), iv=round(iv_u * 100, 1),
            delta=round(g['delta'], 2) if g else None,
            theta=round(g['theta'], 2) if g else None,
            vega=round(g['vega'],  2)  if g else None,
            contracts=contracts, capital=capital, be_pct=be_pct,
        ), best_exp, T
    except Exception:
        return None, None, None


# ── SCAN ONE TICKER ───────────────────────────────────────────────────────────
def scan_ticker(ticker, spy_df, vix, mcap):
    result = dict(
        ticker=ticker, mcap=mcap, error=None,
        rsi_signal=False, momentum_signal=False,
        fund_signal=False, momentum_strong=False,
    )
    try:
        t_obj = yf.Ticker(ticker)
        df    = fix_cols(yf.download(ticker, period='2y', progress=False, auto_adjust=True))
        if len(df) < 210:
            result['error'] = 'insufficient data'; return result

        ind   = calc_indicators(df)
        price = float(ind['c'].iloc[-1])
        rv    = ind['rv']
        curr_iv = float(rv.dropna().iloc[-1]) if not rv.dropna().empty else 0.30
        iv_rank, iv_lo, iv_hi, iv_avg = iv_stats(rv)

        # RSI signal — last RSI_LOOKBACK days
        rsi_series = calc_rsi(ind['c']).dropna()
        rsi_days   = [
            round(float(rsi_series.iloc[-i]), 1) if len(rsi_series) >= i else None
            for i in range(1, RSI_LOOKBACK + 1)
        ]
        rsi_val = rsi_days[0]
        rsi_signal = any(v is not None and v < RSI_THRESHOLD for v in rsi_days)

        # Fundamentals
        try:    info = t_obj.info
        except: info = {}
        grade, fscore, fdetails = score_fundamentals(info)

        # Environmental
        d_earn   = days_to_earnings(t_obj)
        rs_alpha = relative_strength(df, spy_df)

        # Momentum conditions (applied with IV+VIX filter)
        rsi_now   = float(ind['rsi'].iloc[-1])
        s200      = float(ind['sma200'].iloc[-1])
        s50       = float(ind['sma50'].iloc[-1])
        hist_now  = float(ind['macd_hist'].iloc[-1])
        hist_prev = float(ind['macd_hist'].iloc[-2])
        vol_now   = float(ind['v'].iloc[-1])
        vol50     = float(ind['vol50'].iloc[-1])

        mom_cond = dict(
            above_sma200  = price > s200,
            golden_cross  = s50 > s200,
            rsi_neutral   = RSI_MOM_LO <= rsi_now <= RSI_MOM_HI,
            macd_cross_up = hist_now > 0 > hist_prev,
            volume_spike  = vol_now > vol50 * 1.15,
            iv_rank_ok    = iv_rank < IV_RANK_MAX,
            vix_ok        = vix < VIX_MAX,
            earnings_safe = d_earn > MIN_DAYS_EARN,
            rs_positive   = rs_alpha > 0,
            fund_quality  = grade in ('A', 'B'),
            fund_a        = grade == 'A',
        )
        conditions_passed = sum(mom_cond[k] for k in [
            'above_sma200','golden_cross','rsi_neutral','macd_cross_up',
            'volume_spike','iv_rank_ok','vix_ok','earnings_safe','rs_positive',
        ])

        tech_signal     = (mom_cond['above_sma200'] and mom_cond['rsi_neutral'] and
                           mom_cond['macd_cross_up'] and mom_cond['volume_spike'])
        momentum_combo  = tech_signal and mom_cond['fund_quality']
        momentum_strong = momentum_combo and mom_cond['iv_rank_ok'] and mom_cond['vix_ok'] and mom_cond['earnings_safe'] and mom_cond['rs_positive']
        # Momentum signal fires on tech or combo; env filters are advisory but MUST pass for "strong"
        momentum_signal = tech_signal or momentum_combo

        # Fund oversold: Grade A + RSI < 35 (surfaces in oversold section)
        fund_signal = grade == 'A' and rsi_val is not None and rsi_val < 35

        # ── 12M LEAPS ──────────────────────────────────────────────────────
        chain_row, exp_date, T = get_leaps_itm(t_obj, price, curr_iv)
        if chain_row is not None:
            strike   = float(chain_row['strike'])
            bid      = float(chain_row.get('bid') or 0)
            ask      = float(chain_row.get('ask') or 0)
            mid      = (bid + ask) / 2
            last     = float(chain_row.get('lastPrice') or 0)
            prem     = mid if mid > 0 else last
            yahoo_iv = float(chain_row.get('impliedVolatility') or 0)
            solved   = implied_vol(price, strike, T, RISK_FREE_RATE, prem) if prem > 0 else None
            chain_iv = solved or (yahoo_iv if yahoo_iv > 0 else curr_iv)
            greeks   = bs_greeks(price, strike, T, RISK_FREE_RATE, chain_iv)
            if prem <= 0 and greeks: prem = greeks['price']
            if prem <= 0: prem = curr_iv * 0.40 * price
        else:
            T        = T or 1.0
            chain_iv = curr_iv
            exp_date = exp_date or f'+{MIN_LEAPS_DAYS}d (est.)'
            strike   = find_strike_for_delta(price, TARGET_DELTA, T, RISK_FREE_RATE, curr_iv)
            greeks   = bs_greeks(price, strike, T, RISK_FREE_RATE, curr_iv)
            prem     = greeks['price'] if greeks else curr_iv * 0.40 * price

        prem = prem if prem > 0 else curr_iv * 0.40 * price
        contracts, capital = size_position(prem)

        # Trade management levels
        target_option = round(prem * (1 + PROFIT_TARGET_PCT), 2)
        stop_stock    = round(price * (1 - STOP_STOCK_PCT), 2)
        target_stock  = stock_target_for_option_gain(price, strike, T, RISK_FREE_RATE, chain_iv, prem)

        # IV vega-gain estimate (if IV reverts to avg)
        iv_expand_gain = 0.0
        if greeks and greeks.get('vega') and prem > 0:
            iv_expand_gain = round(greeks['vega'] * max(0.0, iv_avg - chain_iv * 100) / prem * 100, 1)

        # ── 3M ATM call ────────────────────────────────────────────────────
        s3, s3_exp, _ = get_3m_option(t_obj, price, curr_iv)

        result.update(dict(
            price=round(price, 2),
            rsi=rsi_val, rsi_days=rsi_days,
            rsi_signal=rsi_signal, fund_signal=fund_signal,
            momentum_signal=momentum_signal, momentum_strong=momentum_strong,
            momentum_combo=momentum_combo, tech_signal=tech_signal,
            mom_cond=mom_cond, conditions_passed=conditions_passed,
            grade=grade, fund_score=fscore, fund_details=fdetails,
            iv_rank=iv_rank, iv_lo=iv_lo, iv_hi=iv_hi, iv_avg=iv_avg,
            curr_iv=round(curr_iv * 100, 1),
            d_earn=d_earn, rs_alpha=round(rs_alpha * 100, 1),
            vix_ok=mom_cond['vix_ok'],
            # 12M LEAPS
            leaps_exp=exp_date, leaps_strike=round(strike, 0),
            leaps_iv=round(chain_iv * 100, 1),
            leaps_entry_sigma=round(chain_iv * 100, 1),
            leaps_prem=round(prem, 2), leaps_contracts=contracts, leaps_capital=capital,
            leaps_delta=round(greeks['delta'], 2) if greeks else None,
            leaps_theta=round(greeks['theta'], 2) if greeks else None,
            leaps_vega=round(greeks['vega'],  2)  if greeks else None,
            leaps_be=round(strike + prem, 2),
            leaps_be_pct=round((strike + prem) / price * 100 - 100, 1),
            iv_expand_gain=iv_expand_gain,
            # Trade levels
            leaps_target_option=target_option,
            leaps_stop_stock=stop_stock,
            leaps_target_stock=target_stock,
            # 3M ATM
            s3=s3, s3_exp=s3_exp or '+3M',
        ))
    except Exception as e:
        result['error'] = str(e)
    return result


# ── CONSOLE OUTPUT ────────────────────────────────────────────────────────────
BAR = '═' * 72

def print_oversold(r, tier):
    rsi_str = ' / '.join(f'{v:.1f}' if v else '—' for v in r['rsi_days'])
    tier_label, _, guidance = tier
    extra = ' [Fund A ★]' if r.get('fund_signal') else ''
    print(f"\n  {'─'*68}")
    print(f"  {r['ticker']:<6}  RSI(last {RSI_LOOKBACK}d)=[{rsi_str}]  "
          f"Grade:{r['grade']}  IV Rank={r['iv_rank']:.0f}/100 [{tier_label}]{extra}")
    print(f"  Price=${r['price']:,.2f}  IV guidance: {guidance}")
    if r.get('leaps_strike'):
        tgt_pct = (r['leaps_target_stock'] / r['price'] - 1) * 100
        print(f"  12M LEAPS: ${r['leaps_strike']:.0f} Call ({r['leaps_exp']})  "
              f"IV={r['leaps_iv']:.0f}%  Prem=${r['leaps_prem']:.2f}  "
              f"BE={r['leaps_be_pct']:+.1f}%  {r['leaps_contracts']}x=${r['leaps_capital']:,.0f}")
        print(f"  Target: stock ≥ ${r['leaps_target_stock']:,.2f} ({tgt_pct:+.1f}%)  "
              f"→ option +50% (${r['leaps_target_option']:.2f})")
        print(f"  Stop  : stock ≤ ${r['leaps_stop_stock']:,.2f} (−10.0%)  "
              f"→ stock-level stop")
        today_str = datetime.now().strftime('%Y-%m-%d')
        print(f"  ┌─ Paste into my_trades.csv ──────────────────────────────────")
        print(f"  │ {r['ticker']},{today_str},{r['price']:.2f},"
              f"{r['leaps_strike']:.0f},C,{r['leaps_exp']},"
              f"{r['leaps_prem']:.2f},{r['leaps_target_stock']:.2f},"
              f"{r['leaps_stop_stock']:.2f},{r['leaps_target_option']:.2f},"
              f"{r['leaps_entry_sigma']},OPEN,,,,,")
        print(f"  └─────────────────────────────────────────────────────────────")
    if r.get('s3'):
        print(f"  3M ATM: ${r['s3']['strike']:.0f} Call ({r['s3_exp']})  "
              f"Prem=${r['s3']['prem']:.2f}  {r['s3']['contracts']}x  BE={r['s3']['be_pct']:+.1f}%")

def print_momentum(r):
    tag = 'STRONG' if r['momentum_strong'] else 'COMBO' if r['momentum_combo'] else 'TECH'
    c   = r['mom_cond']
    checks = [('SMA200',c['above_sma200']),('GldX',c['golden_cross']),('RSI',c['rsi_neutral']),
              ('MACD↑',c['macd_cross_up']),('Vol',c['volume_spike']),
              ('IVR',c['iv_rank_ok']),('VIX',c['vix_ok']),('Earn',c['earnings_safe']),('RS',c['rs_positive'])]
    checks_str = '  '.join(f"{'✓' if v else '✗'}{l}" for l, v in checks)
    print(f"\n  {'─'*68}")
    print(f"  {r['ticker']:<6}  [{tag}]  Grade:{r['grade']}({r['fund_score']}/100)  "
          f"RSI={r['rsi']:.1f}  IV Rank={r['iv_rank']:.0f}  RS={r['rs_alpha']:+.1f}%  Earn={r['d_earn']}d")
    print(f"  {checks_str}")
    if r.get('leaps_strike'):
        print(f"  12M LEAPS: ${r['leaps_strike']:.0f} Call ({r['leaps_exp']})  "
              f"Prem=${r['leaps_prem']:.2f}  BE={r['leaps_be_pct']:+.1f}%  "
              f"{r['leaps_contracts']}x=${r['leaps_capital']:,.0f}")
    if r.get('s3'):
        print(f"  3M ATM: ${r['s3']['strike']:.0f} Call ({r['s3_exp']})  "
              f"Prem=${r['s3']['prem']:.2f}  {r['s3']['contracts']}x  BE={r['s3']['be_pct']:+.1f}%")


# ── EMAIL + TELEGRAM ──────────────────────────────────────────────────────────
def send_telegram(text):
    if not _REQUESTS or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
            data={'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'HTML'},
            timeout=10,
        )
    except Exception as e:
        print(f"  Telegram error: {e}")

def send_email(subject, html_body):
    if not EMAIL_FROM or not EMAIL_PASSWORD:
        print("  Email skipped — set EMAIL_FROM and EMAIL_APP_PASSWORD env vars.")
        return
    try:
        msg            = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = EMAIL_FROM
        msg['To']      = EMAIL_TO
        msg.attach(MIMEText(html_body, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(EMAIL_FROM, EMAIL_PASSWORD)
            s.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print(f"  Email sent → {EMAIL_TO}")
    except Exception as e:
        print(f"  Email error: {e}")


def build_email_html(oversold, momentum, near_oversold, near_mom, vix, run_date):
    BASE = "background:#0d1117;color:#c9d1d9;font-family:'Courier New',monospace;padding:0;margin:0"
    WRAP = "max-width:760px;margin:0 auto;padding:24px"

    def section_header(title, color, count):
        return (f'<div style="color:{color};font-size:13px;font-weight:bold;text-transform:uppercase;'
                f'letter-spacing:1px;margin:24px 0 12px;border-bottom:1px solid {color};padding-bottom:6px">'
                f'{title} &nbsp;<span style="font-size:11px;font-weight:normal;color:#8b949e">'
                f'{count} signal{"s" if count!=1 else ""}</span></div>')

    def leaps_block(r, color, border_color):
        tgt_pct = (r['leaps_target_stock'] / r['price'] - 1) * 100 if r.get('leaps_target_stock') else 0
        iv_rank_tier_label, iv_rank_color, _ = iv_rank_tier(r['iv_rank'])
        gk_str = (f"Δ{r['leaps_delta']} &nbsp;Θ${r['leaps_theta']}/day &nbsp;V${r['leaps_vega']}/1%"
                  if r.get('leaps_delta') else 'N/A')
        iv_expand = (f'<br><span style="color:#3fb950;font-size:11px">+{r["iv_expand_gain"]:.0f}% '
                     f'from vega if IV→avg</span>'
                     if r.get('iv_expand_gain', 0) > 0 else '')
        target_stop = ''
        if r.get('leaps_target_stock'):
            today_str = datetime.now().strftime('%Y-%m-%d')
            csv_row = (f'{r["ticker"]},{today_str},{r["price"]:.2f},'
                       f'{r["leaps_strike"]:.0f},C,{r["leaps_exp"]},'
                       f'{r["leaps_prem"]:.2f},{r["leaps_target_stock"]:.2f},'
                       f'{r["leaps_stop_stock"]:.2f},{r["leaps_target_option"]:.2f},'
                       f'{r["leaps_entry_sigma"]},OPEN,,,,,')
            target_stop = f"""
            <tr>
              <td style="color:#8b949e;padding:4px 0;vertical-align:top">Target&nbsp;/&nbsp;Stop</td>
              <td style="padding:4px 0">
                <b style="color:#3fb950">▲ ${r['leaps_target_stock']:,.2f} ({tgt_pct:+.1f}%)</b>
                <span style="color:#8b949e;font-size:11px"> → option ${r['leaps_target_option']:.2f} (+50%)</span>
                &nbsp;&nbsp;
                <b style="color:#f85149">▼ ${r['leaps_stop_stock']:,.2f} (−10%)</b>
                <br><span style="color:#8b949e;font-size:10px;font-style:italic">
                my_trades.csv: <code>{csv_row}</code></span>
              </td>
            </tr>"""

        s3_block = ''
        if r.get('s3'):
            s3 = r['s3']
            s3_block = f"""
            <tr><td colspan="2" style="padding:10px 0 0">
              <div style="background:#0a0e15;border-left:3px solid #00b0ff;padding:12px;border-radius:4px">
                <div style="color:#00b0ff;font-weight:bold;font-size:12px;margin-bottom:8px">
                  3M ATM CALL &nbsp;<span style="font-weight:normal;color:#8b949e">(momentum / quick-profit)</span>
                </div>
                <span style="color:#8b949e">Strike/Expiry: </span><b>${s3['strike']:.0f} Call ({r['s3_exp']})</b>
                &nbsp;&nbsp;<span style="color:#8b949e">Prem: </span><b>${s3['prem']:.2f}/sh</b>
                &nbsp;&nbsp;<span style="color:#8b949e">IV: </span><b style="color:#00b0ff">{s3['iv']:.0f}%</b>
                &nbsp;&nbsp;<span style="color:#8b949e">Δ</span>{s3.get('delta','~0.50')}
                &nbsp;&nbsp;<b>{s3['contracts']}x = ${s3['capital']:,.0f}</b>
                &nbsp;&nbsp;BE: {s3['be_pct']:+.1f}%
              </div>
            </td></tr>"""

        return f"""
        <div style="background:#161b22;border:1px solid #30363d;border-left:4px solid {border_color};
             border-radius:8px;padding:18px;margin-bottom:14px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
            <div>
              <span style="font-size:20px;font-weight:bold">{r['ticker']}</span>
              &nbsp;<span style="color:#8b949e;font-size:12px">
                Grade {r['grade']}({r['fund_score']}/100) · ${r['price']:,.2f}
                · IV Rank <span style="color:{iv_rank_color}">{r['iv_rank']:.0f}/100 [{iv_rank_tier_label}]</span>
                · {r.get('d_earn','?')}d to earn
              </span>
            </div>
          </div>
          <div style="background:#0d1a0d;border-left:3px solid #238636;border-radius:4px;padding:12px">
            <div style="color:#3fb950;font-weight:bold;font-size:12px;margin-bottom:8px">
              12M LEAPS — Δ≈{TARGET_DELTA} ITM Call
            </div>
            <table style="font-size:13px;border-collapse:collapse;width:100%">
              <tr>
                <td style="color:#8b949e;width:130px;padding:4px 0">Strike / Expiry</td>
                <td style="padding:4px 0"><b>${r['leaps_strike']:.0f} Call</b> ({r['leaps_exp']})</td>
              </tr>
              <tr>
                <td style="color:#8b949e;padding:4px 0">IV (chain)</td>
                <td style="padding:4px 0">
                  <b style="color:{iv_rank_color}">{r['leaps_iv']:.0f}%</b>
                  &nbsp;·&nbsp; 52-wk:
                  <span style="color:#3fb950">{r['iv_lo']:.0f}%</span> →
                  <span style="color:#d29922">{r['iv_avg']:.0f}%</span> →
                  <span style="color:#f85149">{r['iv_hi']:.0f}%</span>
                  {iv_expand}
                </td>
              </tr>
              <tr>
                <td style="color:#8b949e;padding:4px 0">Premium / Greeks</td>
                <td style="padding:4px 0"><b>${r['leaps_prem']:.2f}/sh</b> &nbsp;{gk_str}</td>
              </tr>
              <tr>
                <td style="color:#8b949e;padding:4px 0">Position</td>
                <td style="padding:4px 0">{r['leaps_contracts']}x = <b>${r['leaps_capital']:,.0f}</b>
                  &nbsp; BE: ${r['leaps_be']:,.2f} ({r['leaps_be_pct']:+.1f}%)</td>
              </tr>
              {target_stop}
            </table>
          </div>
          {s3_block}
        </div>"""

    # ── Oversold section ──
    oversold_html = section_header('🔴 Oversold Signals (RSI < 30)', '#f85149', len(oversold))
    if oversold:
        for r in sorted(oversold, key=lambda x: x.get('rsi') or 99):
            tag = '[Fund A ★]' if r.get('fund_signal') else ''
            rsi_str = ' / '.join(f'{v:.1f}' if v else '—' for v in r['rsi_days'])
            oversold_html += f"""
            <div style="color:#8b949e;font-size:12px;margin-bottom:4px;margin-top:2px">
              RSI(last {RSI_LOOKBACK}d): <b style="color:#f85149">[{rsi_str}]</b>
              &nbsp; {tag}
              &nbsp;<span style="color:#8b949e;font-size:11px">
                No IV filter applied — our backtest showed IV filter removes 66–74% win-rate signals</span>
            </div>"""
            oversold_html += leaps_block(r, '#f85149', '#f85149')
    else:
        oversold_html += '<div style="color:#8b949e;padding:16px">No RSI oversold signals today.</div>'

    # ── Momentum section ──
    momentum_html = section_header('🟢 Momentum Signals (RSI 38–62 + Tech + Fund)', '#3fb950', len(momentum))
    if momentum:
        for r in sorted(momentum, key=lambda x: -(4*x.get('momentum_strong',0)+2*x.get('momentum_combo',0))):
            tag = 'STRONG' if r['momentum_strong'] else 'COMBO' if r['momentum_combo'] else 'TECH'
            border = '#00c853' if r['momentum_strong'] else '#00b0ff' if r['momentum_combo'] else '#ffd700'
            c = r['mom_cond']
            cond_html = ' '.join(
                f'<span style="color:{"#3fb950" if ok else "#8b949e"}">'
                f'{"✓" if ok else "·"}{lb}</span>'
                for lb, ok in [
                    ('SMA200',c['above_sma200']),('GldX',c['golden_cross']),
                    ('RSI',c['rsi_neutral']),('MACD↑',c['macd_cross_up']),
                    ('Vol',c['volume_spike']),('IVR',c['iv_rank_ok']),
                    ('VIX',c['vix_ok']),('Earn',c['earnings_safe']),('RS',c['rs_positive']),
                ]
            )
            momentum_html += f"""
            <div style="color:#8b949e;font-size:12px;margin-bottom:4px">
              <span style="background:{border};color:#000;padding:2px 8px;border-radius:6px;
                    font-size:11px;font-weight:bold">{tag}</span>
              &nbsp; RSI={r['rsi']:.1f} &nbsp; RS={r['rs_alpha']:+.1f}% vs SPY &nbsp;
              Earn={r['d_earn']}d &nbsp; {cond_html}
            </div>"""
            momentum_html += leaps_block(r, '#3fb950', border)
    else:
        momentum_html += '<div style="color:#8b949e;padding:16px">No momentum signals today.</div>'

    # ── Near misses ──
    def near_table(items, label):
        if not items: return ''
        rows = ''.join(
            f'<tr><td style="padding:6px 10px;font-weight:bold">{r["ticker"]}</td>'
            f'<td style="padding:6px 10px">{r["grade"]}</td>'
            f'<td style="padding:6px 10px">{r.get("rsi","?"):.1f}</td>'
            f'<td style="padding:6px 10px">{r.get("iv_rank","?"):.0f}/100</td>'
            f'<td style="padding:6px 10px">{r.get("rs_alpha","?"):+.1f}%</td>'
            f'<td style="padding:6px 10px">{r.get("d_earn","?") if isinstance(r.get("d_earn"),int) else "?"}d</td></tr>'
            for r in items[:8]
        )
        return (f'<div style="color:#79c0ff;font-size:11px;text-transform:uppercase;'
                f'letter-spacing:1px;margin:18px 0 8px">{label}</div>'
                f'<div style="overflow-x:auto"><table style="font-size:12px;border-collapse:collapse;'
                f'width:100%;background:#161b22;border:1px solid #30363d;border-radius:6px">'
                f'<thead><tr style="background:#21262d">'
                f'<th style="padding:7px 10px;text-align:left;color:#8b949e;font-weight:normal">Ticker</th>'
                f'<th style="padding:7px 10px;text-align:left;color:#8b949e;font-weight:normal">Grade</th>'
                f'<th style="padding:7px 10px;text-align:left;color:#8b949e;font-weight:normal">RSI</th>'
                f'<th style="padding:7px 10px;text-align:left;color:#8b949e;font-weight:normal">IV Rank</th>'
                f'<th style="padding:7px 10px;text-align:left;color:#8b949e;font-weight:normal">RS vs SPY</th>'
                f'<th style="padding:7px 10px;text-align:left;color:#8b949e;font-weight:normal">Earnings</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div>')

    near_html = near_table(near_oversold, f'Near Oversold — RSI {RSI_THRESHOLD:.0f}–35 (Watch List)')
    near_html += near_table(near_mom, 'Near Momentum — 6+ of 9 conditions met')

    header = (f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
              f'padding:16px 20px;margin-bottom:24px">'
              f'<div style="font-size:18px;font-weight:bold;color:#58a6ff;margin-bottom:4px">'
              f'LEAPS Unified Scanner</div>'
              f'<div style="color:#8b949e;font-size:12px">'
              f'{run_date} &nbsp;·&nbsp; Top {TOP_N} large-caps &nbsp;·&nbsp; '
              f'VIX: <b style="color:{"#3fb950" if vix<20 else "#d29922" if vix<VIX_MAX else "#f85149"}">'
              f'{vix:.1f}</b> &nbsp;·&nbsp; '
              f'<span style="color:#f85149">Oversold: {len(oversold)}</span> &nbsp;·&nbsp; '
              f'<span style="color:#3fb950">Momentum: {len(momentum)}</span>'
              f'</div></div>')

    footer = ('<p style="color:#8b949e;font-size:11px;margin-top:24px;'
              'border-top:1px solid #21262d;padding-top:12px;text-align:center">'
              'LEAPS Unified Scanner · Not financial advice · Data: Yahoo Finance · '
              'Oversold: RSI back-solve IV · Momentum: Newton-Raphson IV</p>')

    return (f'<html><body style="{BASE}"><div style="{WRAP}">'
            f'{header}{oversold_html}{momentum_html}{near_html}{footer}'
            f'</div></body></html>')


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-email', action='store_true', help='Skip email/telegram alerts')
    args = parser.parse_args()

    # Raise OS file-descriptor limit — 40 tickers × ~5 yfinance calls each can
    # exhaust the macOS default of 256. Request 4096; silently ignore if denied.
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 4096), hard))
    except Exception:
        pass

    run_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f'\n{BAR}')
    print(f'  LEAPS UNIFIED SCANNER  —  {run_date}')
    print(f'{BAR}\n  Loading market data...')

    vix_df  = fix_cols(yf.download('^VIX', period='5d', progress=False, auto_adjust=True))
    vix     = float(vix_df['Close'].dropna().iloc[-1])
    spy_df  = fix_cols(yf.download('SPY',  period='2y', progress=False, auto_adjust=True))
    spy_p   = float(spy_df['Close'].iloc[-1])
    spy_s200= float(spy_df['Close'].rolling(200).mean().iloc[-1])
    trend   = 'BULLISH' if spy_p > spy_s200 else 'BEARISH'
    regime  = 'RISK-ON' if vix < 20 else ('ELEVATED' if vix < VIX_MAX else 'RISK-OFF')
    print(f'  VIX={vix:.1f} [{regime}]  SPY={trend}\n')

    tickers, mcap_map = get_large_cap_universe()

    print(f'  Scanning {len(tickers)} tickers...\n')
    results = []
    for ticker in tickers:
        print(f'    {ticker:<6}...', end='', flush=True)
        r = scan_ticker(ticker, spy_df, vix, mcap_map.get(ticker, 0))
        results.append(r)
        if r.get('error'):
            print(f' ERR: {r["error"]}')
        else:
            flags = []
            if r.get('rsi_signal'):   flags.append('OVERSOLD')
            if r.get('fund_signal'):  flags.append('FUND-A')
            if r.get('momentum_strong'): flags.append('STRONG')
            elif r.get('momentum_combo'):flags.append('COMBO')
            elif r.get('momentum_signal'):flags.append('TECH')
            cond_n = r.get('conditions_passed', 0)
            print(f' {" | ".join(flags) if flags else f"({cond_n}/9)"}')

    valid = [r for r in results if not r.get('error')]

    oversold = sorted(
        [r for r in valid if r.get('rsi_signal') or r.get('fund_signal')],
        key=lambda x: x.get('rsi') or 99,
    )
    momentum = sorted(
        [r for r in valid if r.get('momentum_signal')],
        key=lambda x: -(4*x.get('momentum_strong',0) + 2*x.get('momentum_combo',0)),
    )
    near_oversold = sorted(
        [r for r in valid if not (r.get('rsi_signal') or r.get('fund_signal'))
         and r.get('rsi') is not None and RSI_THRESHOLD <= r['rsi'] <= 35],
        key=lambda x: x.get('rsi') or 99,
    )
    near_mom = sorted(
        [r for r in valid if not r.get('momentum_signal') and r.get('conditions_passed', 0) >= 6],
        key=lambda x: -x.get('conditions_passed', 0),
    )

    # ── Print results ──
    print(f'\n{BAR}')
    print(f'  OVERSOLD SIGNALS ({len(oversold)}) — RSI < {RSI_THRESHOLD:.0f}  [no IV filter]')
    print(BAR)
    if oversold:
        for r in oversold:
            tier = iv_rank_tier(r['iv_rank'])
            print_oversold(r, tier)
    else:
        print('  None today.')
        if near_oversold:
            print(f'  Near misses (RSI {RSI_THRESHOLD:.0f}–35): ' +
                  ', '.join(f'{r["ticker"]}({r["rsi"]:.1f})' for r in near_oversold[:5]))

    print(f'\n{BAR}')
    print(f'  MOMENTUM SIGNALS ({len(momentum)}) — RSI 38–62 + Tech + Fund  [IV Rank < {IV_RANK_MAX}]')
    print(BAR)
    if momentum:
        for r in momentum:
            print_momentum(r)
    else:
        print('  None today.')
        if near_mom:
            print(f'  Near misses (6+/9 conditions): ' +
                  ', '.join(f'{r["ticker"]}({r["conditions_passed"]}/9)' for r in near_mom[:5]))

    # ── Save CSV ──
    csv_path = 'leaps_scan_results.csv'
    try:
        rows = [{k: v for k, v in r.items()
                 if k not in ('mom_cond', 'fund_details', 's3', 'rsi_days')}
                for r in valid]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f'\n  Saved → {csv_path}')
    except OSError as e:
        print(f'\n  CSV save failed ({e}) — try: ulimit -n 4096')

    # ── Alerts ──
    if not args.no_email:
        total = len(oversold) + len(momentum)
        tickers_str = ', '.join(
            r['ticker'] for r in sorted(oversold + momentum, key=lambda x: x['ticker'])
        )
        subject = (
            f'LEAPS Scanner {run_date} — {total} signal(s): {tickers_str}'
            if total else
            f'LEAPS Scanner {run_date} — No signals today (VIX={vix:.1f})'
        )
        html = build_email_html(oversold, momentum, near_oversold, near_mom, vix, run_date)
        try:
            with open('leaps_scan.html', 'w') as f:
                f.write(html)
            print('  Saved → leaps_scan.html')
        except OSError as e:
            print(f'  HTML save failed ({e})')

        def _mom_tag(r):
            if r.get('momentum_strong'): return 'STRONG'
            if r.get('momentum_combo'):  return 'COMBO'
            return 'TECH'

        tg_lines = [f'<b>LEAPS Scanner {run_date}</b>  VIX={vix:.1f} [{regime}]']
        if oversold:
            tg_lines.append('\n🔴 <b>Oversold (' + str(len(oversold)) + ')</b>: ' +
                            ', '.join(f'{r["ticker"]}(RSI={r["rsi"]:.0f})' for r in oversold))
        if momentum:
            tg_lines.append('\n🟢 <b>Momentum (' + str(len(momentum)) + ')</b>: ' +
                            ', '.join(f'{r["ticker"]}({_mom_tag(r)})' for r in momentum))
        if not oversold and not momentum:
            tg_lines.append('\nNo signals today.')
        send_telegram('\n'.join(tg_lines))
        send_email(subject, html)

    print(f'\n{BAR}\n')


if __name__ == '__main__':
    main()
