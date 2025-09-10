#!/usr/bin/env python3
"""
Monitor up to 50 tickers for:
  (1) daily price drops vs previous close, and
  (2) "sellout" pressure (seller-initiated dollar volume dominance),
then alert via console (red), sound, email, and write every check to a CSV log.

NEW:
- CSV logging for every run and alert
- Pre-close scans (3:40–4:00 pm ET) at 60s cadence regardless of the normal period
- Cross-sectional comparison for alerts vs. the run-wide averages across the monitored basket

Requires:
  pip install alpaca-py colorama python-dateutil
Env vars (recommended):
  ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, ALPACA_BASE_URL, ALPACA_DATA_FEED
  ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASS
  ALERT_EMAIL_FROM, ALERT_EMAIL_TO
"""

import argparse
import os
import sys
import time
import smtplib
import ssl
import platform
import subprocess
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone

from dateutil import tz
from colorama import init as colorama_init, Fore, Style

# ---- Alpaca SDK v2 ----
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockLatestTradeRequest,
    StockBarsRequest,
    StockTradesRequest,
    StockQuotesRequest,
)
from alpaca.data.timeframe import TimeFrame

import csv
from pathlib import Path

# ------------------ CONFIG ------------------
CONFIG = {
    "drop_threshold_pct": 1.0,        # Trigger if daily drop <= -1.0%
    "sellout_ratio_threshold": 0.62,  # Trigger if seller-initiated $vol / total >= 0.62
    "check_period_minutes": 20,       # Re-check cadence outside pre-close window
    "cooldown_minutes": 60,           # Per-symbol cooldown per alert type
    "min_sellout_dollar_vol": 1_000_000,  # Require at least this $ seller-initiated volume in window
    "trade_window_minutes": 20,       # For sell/buy classification (rolling window)
    "data_feed": os.getenv("ALPACA_DATA_FEED", "iex"),  # "iex" or "sip"
    "email_enabled": True,
    "sound_enabled": True,
    "ny_tz": "America/New_York",
    "log_csv_path": "selloff_monitor_log.csv",          # CSV log
    "preclose_period_seconds": 60,                      # cadence during 15:40–16:00 ET
}

# Email from env or fallbacks
EMAIL = {
    "smtp_host": os.getenv("ALERT_SMTP_HOST", ""),
    "smtp_port": int(os.getenv("ALERT_SMTP_PORT", "587")),
    "smtp_user": os.getenv("ALERT_SMTP_USER", ""),
    "smtp_pass": os.getenv("ALERT_SMTP_PASS", ""),
    "email_from": os.getenv("ALERT_EMAIL_FROM", ""),
    "email_to": os.getenv("ALERT_EMAIL_TO", ""),
}

# Alpaca creds
ALPACA = {
    "key": os.getenv("ALPACA_API_KEY_ID", ""),
    "secret": os.getenv("ALPACA_API_SECRET_KEY", ""),
    "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
}

# --------------------------------------------

def colored(msg, color):
    return color + msg + Style.RESET_ALL

def print_info(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def print_alert(msg):
    print(colored(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", Fore.RED), flush=True)

def play_sound():
    if not CONFIG["sound_enabled"]:
        return
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 400)  # 1kHz, 0.4s
        elif platform.system() == "Darwin":
            subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
        else:
            try:
                subprocess.Popen(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"])
            except Exception:
                sys.stdout.write("\a"); sys.stdout.flush()
    except Exception:
        sys.stdout.write("\a"); sys.stdout.flush()

def send_email(subject, body):
    if not CONFIG["email_enabled"]:
        return
    if not (EMAIL["smtp_host"] and EMAIL["smtp_user"] and EMAIL["smtp_pass"] and EMAIL["email_from"] and EMAIL["email_to"]):
        print_info("Email not configured. Skipping email alert.")
        return
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = EMAIL["email_from"]
        msg["To"] = EMAIL["email_to"]
        context = ssl.create_default_context()
        with smtplib.SMTP(EMAIL["smtp_host"], EMAIL["smtp_port"]) as server:
            server.starttls(context=context)
            server.login(EMAIL["smtp_user"], EMAIL["smtp_pass"])
            server.sendmail(EMAIL["email_from"], [EMAIL["email_to"]], msg.as_string())
    except Exception as e:
        print_info(f"Email send failed: {e}")

def as_ny(dt: datetime) -> datetime:
    tzinfo = tz.gettz(CONFIG["ny_tz"])
    return dt.astimezone(tzinfo)

def in_preclose_window(now_ny=None):
    """3:40 pm to 4:00 pm ET inclusive of 15:40:00 and exclusive of 16:00:00."""
    if now_ny is None:
        now_ny = as_ny(datetime.now(timezone.utc))
    if now_ny.weekday() >= 5:
        return False
    start = now_ny.replace(hour=15, minute=40, second=0, microsecond=0)
    end = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now_ny < end

def build_client():
    if not ALPACA["key"] or not ALPACA["secret"]:
        print_alert("Missing Alpaca API credentials. Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.")
        sys.exit(1)
    return StockHistoricalDataClient(ALPACA["key"], ALPACA["secret"])

def get_prev_close(client, symbol):
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        limit=2,
        feed=CONFIG["data_feed"],
        adjustment="raw",
    )
    bars = client.get_stock_bars(req)
    series = bars.data.get(symbol, [])
    if len(series) < 2:
        return None
    return float(series[-2].close)

def get_latest_trade_price(client, symbol):
    req = StockLatestTradeRequest(symbol_or_symbols=[symbol], feed=CONFIG["data_feed"])
    trades = client.get_stock_latest_trade(req)
    tr = trades.get(symbol)
    if tr is None or tr.trade is None:
        return None
    return float(tr.trade.price)

def fetch_recent_trades_and_quotes(client, symbol, minutes_back=20):
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes_back)
    trq = StockTradesRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
        limit=10_000,
        feed=CONFIG["data_feed"],
    )
    qqr = StockQuotesRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
        limit=10_000,
        feed=CONFIG["data_feed"],
    )
    trades = client.get_stock_trades(trq).data.get(symbol, [])
    quotes = client.get_stock_quotes(qqr).data.get(symbol, [])
    return trades, quotes

def classify_sell_buy_volume(trades, quotes):
    """
    Classify trades via mid-quote:
      trade.price > mid -> buyer-initiated
      trade.price < mid -> seller-initiated
    Returns (sell_dollar_vol, buy_dollar_vol, used_trades).
    """
    if not trades or not quotes:
        return 0.0, 0.0, 0
    qidx = 0
    sell_vol = 0.0
    buy_vol = 0.0
    used = 0
    for tr in trades:
        t_ts = tr.timestamp
        price = float(tr.price)
        size = float(tr.size or 0.0)
        while qidx + 1 < len(quotes) and quotes[qidx + 1].timestamp <= t_ts:
            qidx += 1
        if qidx < len(quotes):
            q = quotes[qidx]
            bid = float(q.bid_price or 0.0)
            ask = float(q.ask_price or 0.0)
            if bid <= 0 or ask <= 0:
                continue
            mid = 0.5 * (bid + ask)
            dollar = price * size
            if price > mid:
                buy_vol += dollar; used += 1
            elif price < mid:
                sell_vol += dollar; used += 1
    return sell_vol, buy_vol, used

def format_pct(x):
    return f"{x:.2f}%"

def ensure_csv_header(path: str):
    p = Path(path)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ts_utc",
            "ts_ny",
            "preclose_window",
            "symbol",
            "price",
            "daily_change_pct",
            "sell_ratio",
            "seller_dollar_vol",
            "buyer_dollar_vol",
            "used_trades",
            "drop_hit",
            "sellout_hit",
            "market_avg_daily_change_pct",
            "market_avg_sell_ratio",
            "rel_daily_change_pct",    # symbol - avg
            "rel_sell_ratio",          # symbol - avg
            "market_shift_flag",       # TRUE if symbol close to avg (both) -> looks market-wide
            "alert_type",              # DROP / SELLOUT / NONE
            "info_msg"                 # text summary
        ])

def append_csv(path: str, row: list):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def monitor_once(client, symbols, state):
    """
    Single pass:
      - compute metrics per symbol
      - compute basket averages
      - print info lines
      - fire alerts with cross-sectional comparison
      - log all to CSV
    """
    now_utc = datetime.now(timezone.utc)
    now_ny = as_ny(now_utc)
    preclose = in_preclose_window(now_ny)
    print_info(
        f"Running check @ {now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')} "
        f"| Feed={CONFIG['data_feed']} | PreClose={preclose}"
    )

    # 1) Collect per-symbol snapshot first so averages reflect same-time view
    snap = []  # list of dicts per symbol for this run
    for sym in symbols:
        rec = {
            "symbol": sym,
            "ok": False,
            "price": None,
            "daily_change_pct": None,
            "sell_ratio": 0.0,
            "sell_dv": 0.0,
            "buy_dv": 0.0,
            "used_trades": 0,
            "err": None
        }
        try:
            prev_close = get_prev_close(client, sym)
            last_price = get_latest_trade_price(client, sym)
            if prev_close is None or last_price is None or prev_close <= 0:
                rec["err"] = "insufficient prev_close/latest"
            else:
                daily_change = (last_price / prev_close - 1.0) * 100.0
                trades, quotes = fetch_recent_trades_and_quotes(client, sym, CONFIG["trade_window_minutes"])
                sell_dv, buy_dv, used_trades = classify_sell_buy_volume(trades, quotes)
                total_dv = sell_dv + buy_dv
                sell_ratio = sell_dv / total_dv if total_dv > 0 else 0.0

                rec.update({
                    "ok": True,
                    "price": last_price,
                    "daily_change_pct": daily_change,
                    "sell_ratio": sell_ratio,
                    "sell_dv": sell_dv,
                    "buy_dv": buy_dv,
                    "used_trades": used_trades
                })
        except Exception as e:
            rec["err"] = str(e)
        snap.append(rec)

    # 2) Compute run-wide averages across symbols with ok=True
    ok_rows = [r for r in snap if r["ok"]]
    avg_change = sum(r["daily_change_pct"] for r in ok_rows) / len(ok_rows) if ok_rows else 0.0
    avg_sell_ratio = sum(r["sell_ratio"] for r in ok_rows) / len(ok_rows) if ok_rows else 0.0

    # 3) Emit info lines + alerts, and log each symbol to CSV
    ensure_csv_header(CONFIG["log_csv_path"])

    for r in snap:
        sym = r["symbol"]
        if not r["ok"]:
            info_msg = f"{sym:>6} | No data ({r['err']})."
            print_info(info_msg)
            append_csv(CONFIG["log_csv_path"], [
                now_utc.isoformat(),
                now_ny.strftime("%Y-%m-%d %H:%M:%S"),
                preclose,
                sym, "", "", "", "", "", "",
                False, False,
                avg_change, avg_sell_ratio,
                "", "", "", "NONE", info_msg
            ])
            continue

        # Info print (non-red)
        info_msg = (
            f"{sym:>6} | Px={r['price']:.2f} | Δ={format_pct(r['daily_change_pct'])} "
            f"| SellRatio={r['sell_ratio']:.2%} | $Sell={r['sell_dv']:,.0f} | $Buy={r['buy_dv']:,.0f} | TradesUsed={r['used_trades']}"
        )
        print_info(info_msg)

        # Decide triggers
        drop_hit = r["daily_change_pct"] <= -CONFIG["drop_threshold_pct"]
        sellout_hit = (r["sell_ratio"] >= CONFIG["sellout_ratio_threshold"]) and (r["sell_dv"] >= CONFIG["min_sellout_dollar_vol"])

        # Cross-sectional deltas
        rel_change = r["daily_change_pct"] - avg_change
        rel_sell = r["sell_ratio"] - avg_sell_ratio

        # Simple flag for market-wide vs idiosyncratic:
        # if both deltas are small, call it a 'market shift'; else 'security-specific'
        market_shift_flag = (abs(rel_change) < 0.20) and (abs(rel_sell) < 0.05)  # 0.20% and 5% points thresholds (tunable)

        # Cooldown bookkeeping
        last_alerts = state.setdefault(sym, {})  # {"drop": dt, "sellout": dt}
        alert_type = "NONE"

        # DROP alert
        if drop_hit:
            from_last = datetime.now(timezone.utc) - last_alerts.get("drop", datetime(1970,1,1, tzinfo=timezone.utc))
            if from_last >= timedelta(minutes=CONFIG["cooldown_minutes"]):
                alert_type = "DROP"
                msg = (
                    f"[DROP] {sym} fell {format_pct(r['daily_change_pct'])} "
                    f"(threshold {CONFIG['drop_threshold_pct']}%). "
                    f"vs Basket Δ={format_pct(avg_change)} "
                    f"(rel={format_pct(rel_change)}) | "
                    f"SellRatio={r['sell_ratio']:.1%} vs Basket={avg_sell_ratio:.1%} "
                    f"(rel={(rel_sell*100):.1f}pp) | "
                    f"{'MARKET-WIDE' if market_shift_flag else 'SECURITY-SPECIFIC'}"
                )
                print_alert(msg)
                play_sound()
                send_email(subject=f"DROP ALERT: {sym}", body=info_msg + "\n\n" + msg)
                last_alerts["drop"] = datetime.now(timezone.utc)

        # SELLOUT alert
        if sellout_hit:
            from_last = datetime.now(timezone.utc) - last_alerts.get("sellout", datetime(1970,1,1, tzinfo=timezone.utc))
            if from_last >= timedelta(minutes=CONFIG["cooldown_minutes"]):
                alert_type = "SELLOUT"
                msg = (
                    f"[SELLOUT] {sym} seller-initiated ratio {r['sell_ratio']:.1%} "
                    f"(≥ {CONFIG['sellout_ratio_threshold']:.0%}), $Sell={r['sell_dv']:,.0f}. "
                    f"Basket SellRatio={avg_sell_ratio:.1%} (rel={(rel_sell*100):.1f}pp). "
                    f"Δ={format_pct(r['daily_change_pct'])} vs Basket Δ={format_pct(avg_change)} "
                    f"(rel={format_pct(rel_change)}) | "
                    f"{'MARKET-WIDE' if market_shift_flag else 'SECURITY-SPECIFIC'}"
                )
                print_alert(msg)
                play_sound()
                send_email(subject=f"SELLOUT ALERT: {sym}", body=info_msg + "\n\n" + msg)
                last_alerts["sellout"] = datetime.now(timezone.utc)

        # CSV log row (for every symbol, every run)
        append_csv(CONFIG["log_csv_path"], [
            now_utc.isoformat(),
            now_ny.strftime("%Y-%m-%d %H:%M:%S"),
            preclose,
            sym,
            f"{r['price']:.4f}",
            f"{r['daily_change_pct']:.4f}",
            f"{r['sell_ratio']:.6f}",
            f"{r['sell_dv']:.2f}",
            f"{r['buy_dv']:.2f}",
            r["used_trades"],
            drop_hit,
            sellout_hit,
            f"{avg_change:.4f}",
            f"{avg_sell_ratio:.6f}",
            f"{rel_change:.4f}",
            f"{rel_sell:.6f}",
            market_shift_flag,
            alert_type,
            info_msg
        ])

def main():
    colorama_init(autoreset=True)

    parser = argparse.ArgumentParser(description="Monitor price drops and sellout pressure via Alpaca Market Data v2, with CSV logging and pre-close scans.")
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated list (max 50)")
    parser.add_argument("--drop", type=float, default=CONFIG["drop_threshold_pct"], help="Drop threshold in percent (default 1.0)")
    parser.add_argument("--sellout", type=float, default=CONFIG["sellout_ratio_threshold"], help="Seller-initiated ratio 0-1 (default 0.62)")
    parser.add_argument("--period", type=int, default=CONFIG["check_period_minutes"], help="Check period in minutes outside pre-close (default 20)")
    parser.add_argument("--no-email", action="store_true", help="Disable email alerts")
    parser.add_argument("--no-sound", action="store_true", help="Disable sound")
    parser.add_argument("--feed", type=str, default=CONFIG["data_feed"], choices=["iex", "sip"], help="Data feed (iex/sip)")
    parser.add_argument("--min-sell-dv", type=float, default=CONFIG["min_sellout_dollar_vol"], help="Minimum $ seller-initiated volume to trigger sellout")
    parser.add_argument("--log", type=str, default=CONFIG["log_csv_path"], help="CSV log path")
    parser.add_argument("--trade-window", type=int, default=CONFIG["trade_window_minutes"], help="Minutes back for trade/quote classification (default 20)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print_alert("No symbols provided."); sys.exit(1)
    if len(symbols) > 50:
        print_alert("Max 50 symbols allowed."); sys.exit(1)

    CONFIG["drop_threshold_pct"] = float(args.drop)
    CONFIG["sellout_ratio_threshold"] = float(args.sellout)
    CONFIG["check_period_minutes"] = int(args.period)
    CONFIG["data_feed"] = args.feed
    CONFIG["min_sellout_dollar_vol"] = float(args.min_sell_dv)
    CONFIG["email_enabled"] = not args.no_email
    CONFIG["sound_enabled"] = not args.no_sound
    CONFIG["log_csv_path"] = args.log
    CONFIG["trade_window_minutes"] = int(args.trade_window)

    client = build_client()

    print_info(f"Monitoring: {', '.join(symbols)}")
    print_info(
        f"Drop≥{CONFIG['drop_threshold_pct']}% | Sellout≥{CONFIG['sellout_ratio_threshold']:.2f} | "
        f"MinSeller$={CONFIG['min_sellout_dollar_vol']:,.0f} | Feed={CONFIG['data_feed']} | "
        f"Period={CONFIG['check_period_minutes']}m | Log={CONFIG['log_csv_path']}"
    )

    # Shared per-symbol alert state for cooldowns
    alert_state = {}
    ensure_csv_header(CONFIG["log_csv_path"])

    # Initial run immediately, then sleep based on pre-close window
    while True:
        monitor_once(client, symbols, alert_state)
        now_ny = as_ny(datetime.now(timezone.utc))
        if in_preclose_window(now_ny):
            time.sleep(CONFIG["preclose_period_seconds"])
        else:
            time.sleep(CONFIG["check_period_minutes"] * 60)

if __name__ == "__main__":
    main()
