#!/usr/bin/env python3
"""
Trade Tracker — monitors open LEAPS positions from my_trades.csv

Usage:
  python trade_tracker.py               # check all open positions, alert if target/stop hit
  python trade_tracker.py --summary     # print summary table only, no alerts
  python trade_tracker.py --report      # save HTML report to trade_report.html

Alerts:
  • macOS desktop notification (osascript)
  • Email to EMAIL_TO when target or stop is crossed for the first time

Setup:
  export EMAIL_FROM="you@gmail.com"
  export EMAIL_APP_PASSWORD="your-app-password"
  export EMAIL_TO="you@gmail.com"
"""

import argparse
import os
import smtplib
import subprocess
import warnings
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings('ignore')

TRADES_CSV   = Path(__file__).parent / 'my_trades.csv'
REPORT_HTML  = Path(__file__).parent / 'trade_report.html'
RISK_FREE_RATE = 0.045

EMAIL_FROM    = os.environ.get('EMAIL_FROM', '')
EMAIL_PASSWORD = os.environ.get('EMAIL_APP_PASSWORD', '')
EMAIL_TO      = os.environ.get('EMAIL_TO', 'avin.khurana18@gmail.com')


# ── Black-Scholes ─────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, option_type='C'):
    if T <= 1e-6 or S <= 0 or K <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == 'C' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    from scipy.stats import norm
    if option_type == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ── Load trades ───────────────────────────────────────────────────────────────
def load_trades():
    """Read my_trades.csv, skip comment lines, return DataFrame of OPEN trades."""
    lines = [l for l in TRADES_CSV.read_text().splitlines()
             if l.strip() and not l.strip().startswith('#')]
    if len(lines) < 2:
        return pd.DataFrame()
    from io import StringIO
    df = pd.read_csv(StringIO('\n'.join(lines)))
    df.columns = df.columns.str.strip()
    df['entry_date']  = pd.to_datetime(df['entry_date'], errors='coerce')
    df['expiry']      = pd.to_datetime(df['expiry'],     errors='coerce')
    for col in ['entry_stock','strike','entry_option','target_stock',
                'stop_stock','target_option','sigma_pct']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def save_trades(df):
    """Write trades back to CSV, preserving comment header block."""
    header_lines = [l for l in TRADES_CSV.read_text().splitlines()
                    if l.strip().startswith('#')]
    cols = ['ticker','entry_date','entry_stock','strike','option_type','expiry',
            'entry_option','target_stock','stop_stock','target_option','sigma_pct',
            'status','exit_date','exit_stock','exit_option','pnl_pct','notes']
    data_lines = df.to_csv(index=False, columns=[c for c in cols if c in df.columns])
    TRADES_CSV.write_text('\n'.join(header_lines) + '\n' + data_lines)


# ── Live prices ───────────────────────────────────────────────────────────────
def fetch_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            p  = getattr(fi, 'last_price', None) or getattr(fi, 'regular_market_price', None)
            if p:
                prices[t] = float(p)
        except Exception:
            pass
    return prices


# ── Notifications ─────────────────────────────────────────────────────────────
def notify_mac(title, message):
    try:
        script = (f'display notification "{message}" with title "{title}" '
                  f'sound name "Glass"')
        subprocess.run(['osascript', '-e', script], capture_output=True, timeout=5)
    except Exception:
        pass


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


def fire_alert(ticker, alert_type, current_stock, row):
    """Send macOS + email alert for a target or stop trigger."""
    if alert_type == 'TARGET':
        icon, color = '🎯', 'green'
        title   = f'TARGET HIT — {ticker}'
        short   = (f'{ticker} reached ${current_stock:.2f} ≥ target ${row.target_stock:.2f}. '
                   f'Consider taking profits on your ${row.strike:.0f}{row.option_type} {row.expiry.strftime("%b %y")} LEAPS.')
    else:
        icon, color = '🛑', 'red'
        title   = f'STOP HIT — {ticker}'
        short   = (f'{ticker} fell to ${current_stock:.2f} ≤ stop ${row.stop_stock:.2f}. '
                   f'Consider exiting your ${row.strike:.0f}{row.option_type} {row.expiry.strftime("%b %y")} LEAPS.')

    notify_mac(title, short)

    html = f"""
    <html><body style="background:#0d1117;color:#c9d1d9;font-family:monospace;padding:24px">
    <div style="max-width:560px;margin:0 auto">
      <div style="background:#161b22;border:2px solid #{'3fb950' if color=='green' else 'f85149'};
           border-radius:8px;padding:20px">
        <div style="font-size:20px;font-weight:bold;color:#{'3fb950' if color=='green' else 'f85149'};
             margin-bottom:12px">{icon} {title}</div>
        <table style="font-size:13px;border-collapse:collapse;width:100%">
          <tr><td style="color:#8b949e;padding:4px 0;width:130px">Ticker</td>
              <td style="padding:4px 0;font-weight:bold;font-size:16px">{ticker}</td></tr>
          <tr><td style="color:#8b949e;padding:4px 0">Current Stock</td>
              <td style="padding:4px 0;color:#{'3fb950' if color=='green' else 'f85149'};font-weight:bold">
                ${current_stock:.2f}</td></tr>
          <tr><td style="color:#8b949e;padding:4px 0">Entry Stock</td>
              <td style="padding:4px 0">${row.entry_stock:.2f}</td></tr>
          <tr><td style="color:#8b949e;padding:4px 0">Contract</td>
              <td style="padding:4px 0">${row.strike:.0f} {row.option_type} exp {row.expiry.strftime('%Y-%m-%d')}</td></tr>
          <tr><td style="color:#8b949e;padding:4px 0">Entry Premium</td>
              <td style="padding:4px 0">${row.entry_option:.2f}/share</td></tr>
          <tr><td style="color:#8b949e;padding:4px 0">{'Target' if alert_type=='TARGET' else 'Stop'}</td>
              <td style="padding:4px 0;font-weight:bold">
                ${'%.2f'%(row.target_stock if alert_type=='TARGET' else row.stop_stock)}</td></tr>
          <tr><td style="color:#8b949e;padding:4px 0">Notes</td>
              <td style="padding:4px 0;color:#8b949e">{row.get('notes','') or '—'}</td></tr>
        </table>
        <div style="margin-top:16px;padding:12px;background:#0d1117;border-radius:4px;
             color:#8b949e;font-size:12px">{short}</div>
      </div>
      <p style="color:#8b949e;font-size:11px;margin-top:12px;text-align:center">
        Trade Tracker · Not financial advice
      </p>
    </div></body></html>"""

    send_email(f"{'🎯' if alert_type=='TARGET' else '🛑'} {title}", html)


# ── Position status ───────────────────────────────────────────────────────────
def evaluate_position(row, current_stock):
    """Compute current option estimate and return status dict."""
    today    = date.today()
    T_remain = max((row.expiry.date() - today).days / 365, 0.001) if pd.notna(row.expiry) else 0.5
    sigma    = (row.sigma_pct / 100) if pd.notna(row.sigma_pct) and row.sigma_pct > 0 else 0.30

    current_option = bs_price(
        current_stock, row.strike, T_remain,
        RISK_FREE_RATE, sigma, row.option_type
    ) if pd.notna(row.strike) else None

    pnl_pct = ((current_option - row.entry_option) / row.entry_option * 100
               if current_option and row.entry_option else None)

    stock_chg = (current_stock - row.entry_stock) / row.entry_stock * 100

    # Determine alert state
    alert = None
    if pd.notna(row.target_stock) and current_stock >= row.target_stock:
        alert = 'TARGET'
    elif pd.notna(row.stop_stock) and current_stock <= row.stop_stock:
        alert = 'STOP'

    return dict(
        current_stock  = current_stock,
        current_option = round(current_option, 2) if current_option else None,
        pnl_pct        = round(pnl_pct, 1) if pnl_pct is not None else None,
        stock_chg_pct  = round(stock_chg, 1),
        days_to_expiry = max((row.expiry.date() - today).days, 0) if pd.notna(row.expiry) else None,
        alert          = alert,
    )


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(df, prices, evaluations):
    bar = '═' * 110
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f'\n{bar}')
    print(f'  Trade Tracker — {now}')
    print(bar)

    open_trades  = df[df['status'] == 'OPEN']
    closed_wins  = df[df['status'] == 'WIN']
    closed_loss  = df[df['status'] == 'LOSS']

    print(f'  Open: {len(open_trades)}  |  Wins: {len(closed_wins)}  |  Losses: {len(closed_loss)}\n')

    if open_trades.empty:
        print('  No open positions.\n')
    else:
        hdr = (f"  {'Ticker':<7} {'Entry':>10} {'Cur Stock':>10} {'Chg%':>6} "
               f"{'Entry Opt':>10} {'Cur Opt':>9} {'P&L%':>7} "
               f"{'Target':>10} {'Stop':>9} {'DTE':>5}  Status")
        print(hdr)
        print('  ' + '─' * 108)
        for _, row in open_trades.iterrows():
            ev   = evaluations.get(row.ticker, {})
            cs   = ev.get('current_stock')
            co   = ev.get('current_option')
            pnl  = ev.get('pnl_pct')
            chg  = ev.get('stock_chg_pct')
            dte  = ev.get('days_to_expiry')
            alrt = ev.get('alert', '')

            cs_str  = f'${cs:.2f}'   if cs  else '—'
            co_str  = f'${co:.2f}'   if co  else '—'
            pnl_str = f'{pnl:+.1f}%' if pnl is not None else '—'
            chg_str = f'{chg:+.1f}%' if chg is not None else '—'
            dte_str = str(dte)        if dte is not None else '—'
            tgt_str = f'${row.target_stock:.2f}' if pd.notna(row.target_stock) else '—'
            stp_str = f'${row.stop_stock:.2f}'   if pd.notna(row.stop_stock)   else '—'

            flag = ''
            if alrt == 'TARGET': flag = '  🎯 TARGET HIT'
            elif alrt == 'STOP': flag = '  🛑 STOP HIT'
            elif pnl and pnl >= 30: flag = '  ↑ near target'
            elif pnl and pnl <= -20: flag = '  ↓ near stop'

            print(f"  {row.ticker:<7} {row.entry_stock:>10.2f} {cs_str:>10} {chg_str:>6} "
                  f"{row.entry_option:>10.2f} {co_str:>9} {pnl_str:>7} "
                  f"{tgt_str:>10} {stp_str:>9} {dte_str:>5}{flag}")

    if not closed_wins.empty or not closed_loss.empty:
        closed = pd.concat([closed_wins, closed_loss])
        print(f'\n  Closed trades ({len(closed)}):')
        for _, row in closed.iterrows():
            pnl  = row.get('pnl_pct', '')
            icon = '✓' if row.status == 'WIN' else '✗'
            print(f"  {icon} {row.ticker:<6}  entered {row.entry_date.strftime('%Y-%m-%d') if pd.notna(row.entry_date) else '?'}"
                  f"  exit {row.get('exit_date','?')}  P&L: {pnl}%  [{row.status}]"
                  f"  {row.get('notes','') or ''}")

    print(f'\n{bar}\n')


# ── HTML report ───────────────────────────────────────────────────────────────
def generate_report(df, evaluations, output_path=REPORT_HTML):
    today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    open_df = df[df['status'] == 'OPEN']

    rows = ''
    for _, row in df.sort_values('status').iterrows():
        ev     = evaluations.get(row.ticker, {})
        cs     = ev.get('current_stock')
        co     = ev.get('current_option')
        pnl    = ev.get('pnl_pct')
        alrt   = ev.get('alert', '')
        status = row.status

        if alrt == 'TARGET':
            border, badge_col = '#3fb950', '#1a3a1a'
            badge = f'<span style="background:{badge_col};color:#3fb950;padding:2px 10px;border-radius:8px;font-size:11px">🎯 TARGET HIT</span>'
        elif alrt == 'STOP':
            border, badge_col = '#f85149', '#3a1a1a'
            badge = f'<span style="background:{badge_col};color:#f85149;padding:2px 10px;border-radius:8px;font-size:11px">🛑 STOP HIT</span>'
        elif status == 'WIN':
            border = '#3fb950'
            badge  = '<span style="background:#1a3a1a;color:#3fb950;padding:2px 10px;border-radius:8px;font-size:11px">WIN</span>'
        elif status == 'LOSS':
            border = '#f85149'
            badge  = '<span style="background:#3a1a1a;color:#f85149;padding:2px 10px;border-radius:8px;font-size:11px">LOSS</span>'
        else:
            pnl_v  = pnl or 0
            border = '#3fb950' if pnl_v > 0 else ('#f85149' if pnl_v < -15 else '#d29922')
            badge  = '<span style="background:#2a2510;color:#d29922;padding:2px 10px;border-radius:8px;font-size:11px">OPEN</span>'

        pnl_col   = '#3fb950' if (pnl or 0) >= 0 else '#f85149'
        cs_str    = f'${cs:.2f}'   if cs  else '—'
        co_str    = f'${co:.2f}'   if co  else '—'
        pnl_str   = f'<span style="color:{pnl_col};font-weight:bold">{pnl:+.1f}%</span>' if pnl is not None else '—'
        tgt_str   = f'${row.target_stock:.2f}' if pd.notna(row.target_stock) else '—'
        stp_str   = f'${row.stop_stock:.2f}'   if pd.notna(row.stop_stock)   else '—'
        dte       = ev.get('days_to_expiry', '')
        notes_str = row.get('notes', '') or ''

        rows += f"""
        <tr style="border-bottom:1px solid #21262d;border-left:3px solid {border}">
          <td style="padding:10px 12px;font-weight:bold;font-size:15px">{row.ticker}</td>
          <td style="padding:10px 12px;color:#8b949e;font-size:12px">
            {row.entry_date.strftime('%Y-%m-%d') if pd.notna(row.entry_date) else '?'}</td>
          <td style="padding:10px 12px">${row.entry_stock:.2f}</td>
          <td style="padding:10px 12px">${row.strike:.0f}{row.option_type} {row.expiry.strftime('%b %y') if pd.notna(row.expiry) else '?'}</td>
          <td style="padding:10px 12px">${row.entry_option:.2f}</td>
          <td style="padding:10px 12px;font-weight:bold">{cs_str}</td>
          <td style="padding:10px 12px">{co_str}</td>
          <td style="padding:10px 12px">{pnl_str}</td>
          <td style="padding:10px 12px;color:#3fb950">{tgt_str}</td>
          <td style="padding:10px 12px;color:#f85149">{stp_str}</td>
          <td style="padding:10px 12px;color:#8b949e">{dte}d</td>
          <td style="padding:10px 12px">{badge}</td>
          <td style="padding:10px 12px;color:#8b949e;font-size:11px">{notes_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>Trade Tracker — {today_str}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0d1117; color:#c9d1d9; font-family:'Courier New',monospace; }}
  .wrap {{ max-width:1400px; margin:0 auto; padding:32px 20px; }}
  h1 {{ font-size:20px; color:#58a6ff; margin-bottom:4px; }}
  .sub {{ color:#8b949e; font-size:12px; margin-bottom:24px; }}
  .tw {{ overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; background:#161b22;
           border:1px solid #30363d; border-radius:8px; overflow:hidden; }}
  thead tr {{ background:#21262d; }}
  th {{ padding:9px 12px; text-align:left; color:#8b949e; font-weight:normal;
        font-size:11px; text-transform:uppercase; letter-spacing:0.5px; white-space:nowrap; }}
  tbody tr:hover {{ background:#1c2128; }}
  .footer {{ color:#8b949e; font-size:11px; margin-top:16px; text-align:center;
             border-top:1px solid #21262d; padding-top:10px; }}
  .cards {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:22px; }}
  .card {{ flex:1; min-width:120px; background:#161b22; border:1px solid #30363d;
           border-radius:8px; padding:14px 16px; }}
  .card-label {{ color:#8b949e; font-size:11px; text-transform:uppercase;
                 letter-spacing:0.7px; margin-bottom:6px; }}
  .card-val {{ font-size:22px; font-weight:bold; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Trade Tracker</h1>
  <div class="sub">{today_str} · auto-refreshes every 5 min · source: my_trades.csv</div>
  <div class="cards">
    <div class="card"><div class="card-label">Open</div>
      <div class="card-val" style="color:#d29922">{len(open_df)}</div></div>
    <div class="card"><div class="card-label">Wins</div>
      <div class="card-val" style="color:#3fb950">{len(df[df['status']=='WIN'])}</div></div>
    <div class="card"><div class="card-label">Losses</div>
      <div class="card-val" style="color:#f85149">{len(df[df['status']=='LOSS'])}</div></div>
    <div class="card"><div class="card-label">Targets Hit</div>
      <div class="card-val" style="color:#3fb950">
        {sum(1 for ev in evaluations.values() if ev.get('alert')=='TARGET')}</div></div>
    <div class="card"><div class="card-label">Stops Hit</div>
      <div class="card-val" style="color:#f85149">
        {sum(1 for ev in evaluations.values() if ev.get('alert')=='STOP')}</div></div>
  </div>
  <div class="tw">
  <table>
    <thead><tr>
      <th>Ticker</th><th>Entry Date</th><th>Entry Stock</th><th>Contract</th>
      <th>Entry Option</th><th>Current Stock</th><th>Cur Option (est.)</th>
      <th>P&amp;L%</th><th>Target Stock</th><th>Stop Stock</th>
      <th>DTE</th><th>Status</th><th>Notes</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
  <p class="footer">
    Trade Tracker · Not financial advice · Option estimates via Black-Scholes (const. IV) ·
    P&amp;L shown on option price, stop on stock price
  </p>
</div>
</body>
</html>"""

    output_path.write_text(html)
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='LEAPS Trade Position Tracker')
    parser.add_argument('--summary', action='store_true', help='Print summary only, skip alerts')
    parser.add_argument('--report',  action='store_true', help='Save HTML report')
    args = parser.parse_args()

    if not TRADES_CSV.exists():
        print(f'  No trades file found at {TRADES_CSV}')
        print('  Create my_trades.csv and add your positions.')
        return

    df = load_trades()
    if df.empty:
        print('  No trades found in my_trades.csv (only comments or empty file).')
        return

    open_trades = df[df['status'] == 'OPEN']
    if open_trades.empty:
        print('  No OPEN positions to track.')
        return

    # Fetch live prices for all open tickers
    tickers = open_trades['ticker'].unique().tolist()
    print(f'\n  Fetching live prices for {len(tickers)} open position(s): {", ".join(tickers)}...')
    prices = fetch_prices(tickers)

    # Evaluate each open position
    evaluations = {}
    alerts_to_fire = []

    for _, row in open_trades.iterrows():
        ticker = row.ticker
        if ticker not in prices:
            print(f'  {ticker}: could not fetch price — skipped')
            continue
        ev = evaluate_position(row, prices[ticker])
        evaluations[ticker] = ev

        if ev['alert'] and not args.summary:
            alerts_to_fire.append((ticker, ev['alert'], prices[ticker], row))

    # Print summary
    print_summary(df, prices, evaluations)

    # Fire alerts
    if alerts_to_fire:
        print(f'  Firing {len(alerts_to_fire)} alert(s)...')
        for ticker, alert_type, current_stock, row in alerts_to_fire:
            print(f'    {"🎯" if alert_type=="TARGET" else "🛑"} {ticker}: {alert_type} at ${current_stock:.2f}')
            fire_alert(ticker, alert_type, current_stock, row)
    elif not args.summary:
        print('  No target/stop alerts to fire.\n')

    # HTML report
    if args.report:
        path = generate_report(df, evaluations)
        print(f'  Report saved → {path}')
        subprocess.run(['open', str(path)], capture_output=True)


if __name__ == '__main__':
    main()
