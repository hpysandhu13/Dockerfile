"""
╔══════════════════════════════════════════════════════════════╗
║     INSTITUTIONAL SIGNAL INTELLIGENCE BOT v3.0              ║
║     Smart Money Confluence Engine — WebSocket Live Feed      ║
║     Strategy: 200 EMA + RSI(14) + Volume Spread Analysis     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import ccxt
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO
import psycopg2
from psycopg2.extras import RealDictCursor

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("SignalBot")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
WATCHLIST = {
    "crypto": [
        {"id": "BTC/USDT",  "label": "Bitcoin",    "exchange": "binance"},
        {"id": "ETH/USDT",  "label": "Ethereum",   "exchange": "binance"},
        {"id": "SOL/USDT",  "label": "Solana",     "exchange": "binance"},
        {"id": "BNB/USDT",  "label": "BNB",        "exchange": "binance"},
        {"id": "XRP/USDT",  "label": "XRP",        "exchange": "binance"},
        {"id": "AVAX/USDT", "label": "Avalanche",  "exchange": "binance"},
    ],
    "forex": [
        {"id": "EURUSD=X",  "label": "EUR/USD"},
        {"id": "GBPUSD=X",  "label": "GBP/USD"},
        {"id": "USDJPY=X",  "label": "USD/JPY"},
        {"id": "AUDUSD=X",  "label": "AUD/USD"},
    ],
    "commodity": [
        {"id": "GC=F",      "label": "Gold"},
        {"id": "SI=F",      "label": "Silver"},
        {"id": "CL=F",      "label": "Crude Oil"},
        {"id": "NG=F",      "label": "Nat. Gas"},
        {"id": "PL=F",      "label": "Platinum"},
    ],
}

# Strategy params
EMA_PERIOD       = 200
RSI_PERIOD       = 14
RSI_OVERSOLD     = 30
RSI_OVERBOUGHT   = 70
VOLUME_LOOKBACK  = 10
VOLUME_THRESHOLD = 1.50   # 50 % above avg
ATR_MULTIPLIER   = 1.5
MIN_CANDLES      = 220    # need at least EMA_PERIOD + buffer

REFRESH_SECONDS  = 60   # Full signal / indicator cycle interval
EMIT_SECONDS     = 1    # WebSocket emission cadence

# ─────────────────────────────────────────────
#  DATABASE  (optional – bot runs without it)
# ─────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")

def get_db_conn():
    if not DATABASE_URL:
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        log.warning(f"DB connect failed: {e}")
        return None

def init_db():
    conn = get_db_conn()
    if not conn:
        log.warning("No DATABASE_URL set – running without signal persistence.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id          SERIAL PRIMARY KEY,
                    ts          TIMESTAMPTZ DEFAULT NOW(),
                    asset       TEXT,
                    asset_class TEXT,
                    signal      TEXT,
                    price       NUMERIC,
                    entry_low   NUMERIC,
                    entry_high  NUMERIC,
                    stop_loss   NUMERIC,
                    rsi         NUMERIC,
                    ema200      NUMERIC
                )
            """)
        conn.commit()
        log.info("DB initialised.")
    except Exception as e:
        log.warning(f"DB init error: {e}")
    finally:
        conn.close()

def log_signal_to_db(row: dict):
    conn = get_db_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals
                  (asset, asset_class, signal, price, entry_low, entry_high, stop_loss, rsi, ema200)
                VALUES (%(asset)s, %(asset_class)s, %(signal)s, %(price)s,
                        %(entry_low)s, %(entry_high)s, %(stop_loss)s, %(rsi)s, %(ema200)s)
            """, row)
        conn.commit()
    except Exception as e:
        log.warning(f"DB insert error: {e}")
    finally:
        conn.close()

# ─────────────────────────────────────────────
#  TECHNICAL INDICATORS  (pure numpy/pandas)
# ─────────────────────────────────────────────
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> float:
    delta  = series.diff().dropna()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l  = loss.ewm(com=period - 1, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    rsi_s  = 100 - (100 / (1 + rs))
    return round(float(rsi_s.iloc[-1]), 2)

def atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(com=period - 1, adjust=False).mean().iloc[-1])

def volume_spike(df: pd.DataFrame) -> bool:
    """True if latest volume > 150 % of the prior 10-candle average."""
    if len(df) < VOLUME_LOOKBACK + 1:
        return False
    recent_avg = df["Volume"].iloc[-(VOLUME_LOOKBACK + 1):-1].mean()
    if recent_avg == 0:
        return False
    return float(df["Volume"].iloc[-1]) >= recent_avg * VOLUME_THRESHOLD

# ─────────────────────────────────────────────
#  SIGNAL LOGIC
# ─────────────────────────────────────────────
def compute_signal(df: pd.DataFrame, label: str, asset_class: str) -> dict:
    """
    Returns a signal dict.  Never raises – all errors produce NEUTRAL.
    Entry zone and stop loss are always populated when price data is available.
    """
    base = {
        "asset": label,
        "asset_class": asset_class.upper(),
        "price": None,
        "signal": "NEUTRAL",
        "entry_low": None,
        "entry_high": None,
        "stop_loss": None,
        "rsi": None,
        "ema200": None,
        "error": None,
    }

    try:
        if df is None or len(df) < MIN_CANDLES:
            base["error"] = "Insufficient data"
            return base

        df = df.copy()
        price      = float(df["Close"].iloc[-1])
        ema200_val = float(ema(df["Close"], EMA_PERIOD).iloc[-1])
        rsi_val    = rsi(df["Close"], RSI_PERIOD)
        atr_val    = atr(df)
        vol_spike  = volume_spike(df)

        base["price"]  = round(price, 5)
        base["rsi"]    = rsi_val
        base["ema200"] = round(ema200_val, 5)

        sl_distance = ATR_MULTIPLIER * atr_val

        # ── Always show entry zone and ATR-based stop loss ──────────────
        base["entry_low"]  = round(price * 0.999, 5)
        base["entry_high"] = round(price * 1.001, 5)
        # Default stop is long-side (below price); overridden for SELL signals
        base["stop_loss"]  = round(price - sl_distance, 5)

        # ── STRONG BUY confluences ──────────────────────────────────────
        if price > ema200_val and rsi_val < RSI_OVERSOLD and vol_spike:
            base["signal"] = "STRONG BUY ▲"

        # ── STRONG SELL confluences ─────────────────────────────────────
        elif price < ema200_val and rsi_val > RSI_OVERBOUGHT and vol_spike:
            base["signal"]   = "STRONG SELL ▼"
            base["stop_loss"] = round(price + sl_distance, 5)

    except Exception as e:
        log.warning(f"Signal compute error [{label}]: {e}")
        base["error"] = str(e)

    return base

# ─────────────────────────────────────────────
#  DATA FETCHERS
# ─────────────────────────────────────────────
def _apply_realtime_price(result: dict, rt: float, atr_val: float) -> None:
    """Override price and recalculate entry/stop around the live price in-place."""
    sl_distance = ATR_MULTIPLIER * atr_val
    result["price"]      = round(rt, 5)
    if result.get("entry_low") is not None:
        result["entry_low"]  = round(rt * 0.999, 5)
        result["entry_high"] = round(rt * 1.001, 5)
        if "SELL" in result["signal"]:
            result["stop_loss"] = round(rt + sl_distance, 5)
        else:
            result["stop_loss"] = round(rt - sl_distance, 5)


def _yf_realtime_price(ticker: str) -> Optional[float]:
    """Return the live/current price for a yfinance ticker, or None on failure."""
    try:
        fi = yf.Ticker(ticker).fast_info
        price = getattr(fi, "last_price", None)
        if price is None or price == 0:
            price = getattr(fi, "regular_market_price", None)
        if price and float(price) > 0:
            return float(price)
    except Exception as e:
        log.debug(f"fast_info unavailable for {ticker}: {e}")
    return None


def fetch_yfinance(ticker: str, label: str, asset_class: str) -> dict:
    try:
        raw = yf.download(
            ticker,
            period="2y",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if raw is None or raw.empty or len(raw) < MIN_CANDLES:
            log.warning(f"yfinance: no/insufficient data for {ticker}")
            return {"asset": label, "asset_class": asset_class.upper(),
                    "signal": "NEUTRAL", "price": None, "rsi": None,
                    "ema200": None, "entry_low": None, "entry_high": None,
                    "stop_loss": None, "error": "Market closed or no data"}

        # yfinance MultiIndex fix
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        result = compute_signal(raw, label, asset_class)

        # Override price with live quote (more accurate than yesterday's close)
        rt = _yf_realtime_price(ticker)
        if rt is not None:
            _apply_realtime_price(result, rt, atr(raw))

        return result

    except Exception as e:
        log.warning(f"yfinance fetch error [{ticker}]: {e}")
        return {"asset": label, "asset_class": asset_class.upper(),
                "signal": "NEUTRAL", "price": None, "rsi": None,
                "ema200": None, "entry_low": None, "entry_high": None,
                "stop_loss": None, "error": str(e)}


def fetch_ccxt(symbol: str, label: str, exchange_id: str = "binance") -> dict:
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})

        # Fetch 300 daily candles (plenty for 200 EMA)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1d", limit=300)
        if not ohlcv or len(ohlcv) < MIN_CANDLES:
            return {"asset": label, "asset_class": "CRYPTO",
                    "signal": "NEUTRAL", "price": None, "rsi": None,
                    "ema200": None, "entry_low": None, "entry_high": None,
                    "stop_loss": None, "error": "Insufficient OHLCV data"}

        df = pd.DataFrame(ohlcv, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        df = df.astype(float)

        result = compute_signal(df, label, "CRYPTO")

        # Override price with live ticker (real-time, not last candle close)
        try:
            ticker_data = exchange.fetch_ticker(symbol)
            rt = ticker_data.get("last")
            if rt and float(rt) > 0:
                _apply_realtime_price(result, float(rt), atr(df))
        except Exception as te:
            log.debug(f"ccxt fetch_ticker failed for {symbol}: {te}")

        return result

    except Exception as e:
        log.warning(f"ccxt fetch error [{symbol}]: {e}")
        return {"asset": label, "asset_class": "CRYPTO",
                "signal": "NEUTRAL", "price": None, "rsi": None,
                "ema200": None, "entry_low": None, "entry_high": None,
                "stop_loss": None, "error": str(e)}

# ─────────────────────────────────────────────
#  SIGNAL RUNNER  (threaded, non-blocking)
# ─────────────────────────────────────────────
_signal_cache: list  = []
_last_updated: str   = "—"
_cache_lock          = threading.Lock()
_prev_prices: dict   = {}   # { asset_label: last_emitted_price }

def run_all_signals():
    results = []

    for item in WATCHLIST["crypto"]:
        results.append(fetch_ccxt(item["id"], item["label"], item["exchange"]))

    for item in WATCHLIST["forex"]:
        results.append(fetch_yfinance(item["id"], item["label"], "FOREX"))

    for item in WATCHLIST["commodity"]:
        results.append(fetch_yfinance(item["id"], item["label"], "COMMODITY"))

    # Log actionable signals to DB
    for r in results:
        if "BUY" in r["signal"] or "SELL" in r["signal"]:
            log_signal_to_db({
                "asset":       r["asset"],
                "asset_class": r["asset_class"],
                "signal":      r["signal"],
                "price":       r.get("price"),
                "entry_low":   r.get("entry_low"),
                "entry_high":  r.get("entry_high"),
                "stop_loss":   r.get("stop_loss"),
                "rsi":         r.get("rsi"),
                "ema200":      r.get("ema200"),
            })

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with _cache_lock:
        global _signal_cache, _last_updated, _prev_prices
        for r in results:
            label      = r["asset"]
            curr_price = r.get("price")
            prev_price = _prev_prices.get(label)
            if curr_price is not None and prev_price is not None and prev_price != 0:
                r["change_pct"] = round((curr_price - prev_price) / prev_price * 100, 4)
            else:
                r["change_pct"] = 0.0
            if curr_price is not None:
                _prev_prices[label] = curr_price
        _signal_cache  = results
        _last_updated  = ts

    log.info(f"Signal cycle complete — {ts}")

def background_loop():
    while True:
        try:
            run_all_signals()
        except Exception as e:
            log.error(f"Unhandled error in signal loop: {e}")
        time.sleep(REFRESH_SECONDS)

def emission_loop():
    """Push the latest cached data to all connected WebSocket clients every second."""
    while True:
        try:
            with _cache_lock:
                data = {
                    "last_updated": _last_updated,
                    "signals":      list(_signal_cache),
                }
            if data["signals"]:
                socketio.emit("price_update", data)
        except Exception as e:
            log.error(f"Emit error: {e}")
        time.sleep(EMIT_SECONDS)

# ─────────────────────────────────────────────
#  HTML DASHBOARD  (Bloomberg Midnight Theme)
# ─────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SIGNALIQ — Institutional Live Feed</title>
<style>
  /* ── Bloomberg Midnight Theme ─────────────── */
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:     #000000;
    --panel:  #060606;
    --border: #1a1a1a;
    --green:  #00ff9d;
    --red:    #ff3d71;
    --cyan:   #00e5ff;
    --amber:  #ffb700;
    --white:  #ffffff;
    --muted:  #555555;
    --text:   #cccccc;
    --font:   'Courier New', Courier, monospace;
  }
  html, body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
    min-height: 100vh;
  }
  /* ── Header ───────────────────────────────── */
  .hdr {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .logo {
    font-size: 20px;
    font-weight: bold;
    letter-spacing: 5px;
    color: var(--cyan);
    text-transform: uppercase;
  }
  .logo span { color: var(--white); }
  .tagline { font-size: 9px; color: var(--muted); letter-spacing: 2px; margin-top: 2px; }
  .hdr-right { display: flex; gap: 24px; align-items: center; }
  .stat-item { display: flex; flex-direction: column; align-items: flex-end; }
  .stat-label { font-size: 9px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }
  .stat-val   { color: var(--cyan); font-weight: bold; font-size: 12px; }
  /* ── WS status dot ────────────────────────── */
  .ws-status { display: flex; align-items: center; gap: 6px; font-size: 10px; }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
  .dot.live  { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .dot.dead  { background: var(--red);   box-shadow: 0 0 8px var(--red); }
  /* ── Strategy bar ─────────────────────────── */
  .strategy-bar {
    padding: 4px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    background: #040404;
  }
  .badge {
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 1px 7px;
    border: 1px solid #1a1a1a;
  }
  /* ── Table ────────────────────────────────── */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; }
  thead { background: #040404; position: sticky; top: 0; z-index: 10; }
  th {
    padding: 5px 10px;
    text-align: left;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--cyan);
    white-space: nowrap;
    font-weight: normal;
  }
  tr.data-row { border-bottom: 1px solid var(--border); }
  tr.data-row:hover { background: #0a0a0a; }
  td { padding: 5px 10px; vertical-align: middle; white-space: nowrap; }
  tr.sec-hdr td {
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 10px;
    color: var(--muted);
    background: #040404;
    border-top: 1px solid var(--border);
  }
  /* ── Asset name ───────────────────────────── */
  .asset-name { color: var(--white); font-weight: bold; font-size: 13px; }
  /* ── Class tags ───────────────────────────── */
  .tag { font-size: 9px; padding: 1px 5px; letter-spacing: 1px; text-transform: uppercase; border: 1px solid; }
  .tag-crypto    { color: #a78bfa; border-color: #3d2f8a; }
  .tag-forex     { color: #38bdf8; border-color: #1a4a6a; }
  .tag-commodity { color: var(--amber); border-color: #5a4000; }
  /* ── Price cell ───────────────────────────── */
  .price-cell {
    color: var(--white);
    font-weight: bold;
    font-size: 14px;
    display: inline-block;
    min-width: 90px;
  }
  /* ── Change % ─────────────────────────────── */
  .chg-up   { color: var(--green); }
  .chg-down { color: var(--red); }
  .chg-flat { color: var(--muted); }
  /* ── Signal pill ──────────────────────────── */
  .sig {
    font-size: 10px;
    font-weight: bold;
    padding: 2px 8px;
    letter-spacing: 1px;
    display: inline-block;
    border: 1px solid;
  }
  .sig-buy  { color: var(--green); border-color: #00ff9d44; background: #001f10; }
  .sig-sell { color: var(--red);   border-color: #ff3d7144; background: #200010; }
  .sig-neu  { color: var(--muted); border-color: #1a1a1a;   background: #040404; }
  /* ── RSI chip ─────────────────────────────── */
  .rsi-chip { font-size: 11px; padding: 1px 6px; border: 1px solid var(--border); }
  .rsi-low  { color: var(--green); border-color: #00ff9d44; background: #001f10; }
  .rsi-high { color: var(--red);   border-color: #ff3d7144; background: #200010; }
  /* ── Sparkline canvas ─────────────────────── */
  .spark { display: block; }
  /* ── Flash animations ─────────────────────── */
  @keyframes flashUp {
    0%   { background: rgba(0,255,157,0.30); color: #00ff9d; }
    100% { background: transparent;          color: #ffffff; }
  }
  @keyframes flashDown {
    0%   { background: rgba(255,61,113,0.30); color: #ff3d71; }
    100% { background: transparent;           color: #ffffff; }
  }
  .flash-up   { animation: flashUp   0.7s ease-out forwards; }
  .flash-down { animation: flashDown 0.7s ease-out forwards; }
  /* ── Footer ───────────────────────────────── */
  footer {
    padding: 5px 16px;
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 1px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
  }
  .spinner {
    display: inline-block;
    width: 8px; height: 8px;
    border: 1px solid var(--muted);
    border-top-color: var(--cyan);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 4px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); }
</style>
</head>
<body>

<div class="hdr">
  <div>
    <div class="logo">SIGNAL<span>IQ</span></div>
    <div class="tagline">Institutional Smart Money Confluence Engine · WebSocket Live Feed</div>
  </div>
  <div class="hdr-right">
    <div class="stat-item">
      <span class="stat-label">Strategy</span>
      <span class="stat-val" style="font-size:10px;">EMA200 · RSI14 · VSA</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Last Updated</span>
      <span class="stat-val" id="ts">—</span>
    </div>
    <div class="ws-status">
      <div class="dot" id="dot"></div>
      <span id="ws-label" style="color:var(--muted);">CONNECTING</span>
    </div>
  </div>
</div>

<div class="strategy-bar">
  <span class="badge">EMA 200 Trend Filter</span>
  <span class="badge">RSI &lt;30 / &gt;70 Extreme</span>
  <span class="badge">Vol Spike ≥ 150% Avg</span>
  <span class="badge">Stop = 1.5× ATR</span>
  <span class="badge">15 Assets Live</span>
</div>

<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th>Asset</th>
      <th>Class</th>
      <th>Price</th>
      <th>Change&nbsp;%</th>
      <th>Signal</th>
      <th>Trend (30T)</th>
      <th>RSI(14)</th>
      <th>Entry Zone</th>
      <th>Stop Loss</th>
    </tr>
  </thead>
  <tbody id="tbody">
    <tr><td colspan="9" style="text-align:center;padding:50px;color:#555;">
      <span class="spinner"></span>Connecting to live WebSocket feed...
    </td></tr>
  </tbody>
</table>
</div>

<footer>
  <span>SIGNALIQ · FOR INFORMATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE</span>
  <span id="tick-ctr">TICKS: 0</span>
</footer>

<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script>
// ── Constants ──────────────────────────────────────────────────────────
const MAX_PTS = 30;

// Stable display order and section breaks
const ASSET_ORDER = [
  'Bitcoin','Ethereum','Solana','BNB','XRP','Avalanche',
  'EUR/USD','GBP/USD','USD/JPY','AUD/USD',
  'Gold','Silver','Crude Oil','Nat. Gas','Platinum'
];
const SECTION_START = { 'Bitcoin':'CRYPTO', 'EUR/USD':'FOREX', 'Gold':'COMMODITY' };

// ── Per-asset client-side state ────────────────────────────────────────
const priceHistory = {};   // { label: [p1, p2, ...] }
const prevPrices   = {};   // { label: lastSeenPrice }

let tickCount   = 0;
let tableBuilt  = false;

// ── Decimal places per asset ───────────────────────────────────────────
function pDec(asset) {
  if (['Bitcoin','Ethereum'].includes(asset))            return 2;
  if (['BNB','Solana','Avalanche'].includes(asset))      return 2;
  if (['Gold','Silver','Platinum'].includes(asset))      return 2;
  if (['Crude Oil','Nat. Gas'].includes(asset))          return 3;
  if (['XRP'].includes(asset))                           return 4;
  return 5;
}

// ── Formatting helpers ─────────────────────────────────────────────────
function fmt(v, d) { return v != null ? Number(v).toFixed(d) : '—'; }

function classTag(cls) {
  const map = { CRYPTO:['crypto','Crypto'], FOREX:['forex','Forex'], COMMODITY:['commodity','Cmdty'] };
  const [c, l] = map[cls] || ['forex', cls];
  return `<span class="tag tag-${c}">${l}</span>`;
}

function signalTag(sig) {
  if (!sig)               return `<span class="sig sig-neu">NEUTRAL</span>`;
  if (sig.includes('BUY'))  return `<span class="sig sig-buy">${sig}</span>`;
  if (sig.includes('SELL')) return `<span class="sig sig-sell">${sig}</span>`;
  return `<span class="sig sig-neu">NEUTRAL</span>`;
}

function rsiTag(v) {
  if (v == null) return '<span class="rsi-chip">—</span>';
  const cls = v < 30 ? 'rsi-low' : v > 70 ? 'rsi-high' : '';
  return `<span class="rsi-chip ${cls}">${v}</span>`;
}

function chgTag(pct) {
  if (pct == null || pct === 0) return `<span class="chg-flat">—</span>`;
  const sign  = pct > 0 ? '+' : '';
  const cls   = pct > 0 ? 'chg-up' : 'chg-down';
  const arrow = pct > 0 ? '▲' : '▼';
  return `<span class="${cls}">${sign}${pct.toFixed(4)}%&nbsp;${arrow}</span>`;
}

// ── Glow / flash effect ────────────────────────────────────────────────
function flashCell(cell, dir) {
  cell.classList.remove('flash-up', 'flash-down');
  void cell.offsetWidth;   // trigger reflow so the animation restarts cleanly
  cell.classList.add(dir === 'up' ? 'flash-up' : 'flash-down');
}

// ── Canvas sparkline ───────────────────────────────────────────────────
function drawSparkline(canvas, data) {
  if (!canvas || !data || data.length < 2) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const min   = Math.min(...data);
  const max   = Math.max(...data);
  const range = (max - min) || (min * 0.001) || 1;
  const isUp  = data[data.length - 1] >= data[data.length - 2];
  const color = isUp ? '#00ff9d' : '#ff3d71';

  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 4;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = (h - 2) - ((v - min) / range) * (h - 4);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ── Row ID helper ──────────────────────────────────────────────────────
function rowId(asset) { return 'row-' + asset.replace(/[^a-z0-9]/gi, '_'); }
function pcId(asset)  { return 'pc-'  + asset.replace(/[^a-z0-9]/gi, '_'); }

// ── Build initial table ────────────────────────────────────────────────
function buildTable(signals) {
  const byAsset = {};
  signals.forEach(r => { byAsset[r.asset] = r; });

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';

  ASSET_ORDER.forEach(name => {
    const r = byAsset[name];
    if (!r) return;

    if (SECTION_START[name]) {
      const sh = document.createElement('tr');
      sh.className = 'sec-hdr';
      sh.innerHTML = `<td colspan="9">${SECTION_START[name]}</td>`;
      tbody.appendChild(sh);
    }

    const tr = document.createElement('tr');
    tr.id        = rowId(name);
    tr.className = 'data-row';
    tr.innerHTML = buildCells(r, 'flat');
    tbody.appendChild(tr);

    if (r.price != null) {
      priceHistory[name] = [r.price];
      prevPrices[name]   = r.price;
    }
  });

  tableBuilt = true;
}

// ── Build all <td> content for a row ──────────────────────────────────
function buildCells(r, dir) {
  const d    = pDec(r.asset);
  const zone = (r.entry_low != null && r.entry_high != null)
    ? `${fmt(r.entry_low, d)} – ${fmt(r.entry_high, d)}`
    : '—';
  const sl  = r.stop_loss != null
    ? `<span style="color:var(--red);">${fmt(r.stop_loss, d)}</span>`
    : '—';
  const err = r.error
    ? `<span style="font-size:9px;color:#ff6b6b;display:block;">⚠ ${r.error}</span>`
    : '';

  return `
    <td><span class="asset-name">${r.asset}</span>${err}</td>
    <td>${classTag(r.asset_class)}</td>
    <td><span class="price-cell" id="${pcId(r.asset)}">${r.price != null ? fmt(r.price, d) : '—'}</span></td>
    <td>${chgTag(r.change_pct)}</td>
    <td>${signalTag(r.signal)}</td>
    <td><canvas class="spark" width="80" height="26"></canvas></td>
    <td>${rsiTag(r.rsi)}</td>
    <td style="font-size:11px;">${zone}</td>
    <td style="font-size:11px;">${sl}</td>
  `;
}

// ── Update a single row in-place ───────────────────────────────────────
function updateRow(r) {
  const name = r.asset;
  const row  = document.getElementById(rowId(name));
  if (!row) return;

  const d         = pDec(name);
  const currPrice = r.price;
  const prev      = prevPrices[name];
  let   dir       = 'flat';
  if (currPrice != null && prev != null) {
    if      (currPrice > prev) dir = 'up';
    else if (currPrice < prev) dir = 'down';
  }

  // Append to sparkline history only when price actually changes
  if (currPrice != null) {
    if (!priceHistory[name]) priceHistory[name] = [];
    const hist = priceHistory[name];
    if (hist.length === 0 || hist[hist.length - 1] !== currPrice) {
      hist.push(currPrice);
      if (hist.length > MAX_PTS) hist.shift();
    }
    prevPrices[name] = currPrice;
  }

  // Update price cell with flash
  const pc = document.getElementById(pcId(name));
  if (pc && currPrice != null) {
    const formatted = fmt(currPrice, d);
    if (pc.textContent !== formatted) {
      pc.textContent = formatted;
      if (dir !== 'flat') flashCell(pc, dir);
    }
  }

  // Update Change %, Signal, RSI, Entry Zone, Stop Loss via cell index (stable column order)
  const cells = row.cells;
  if (cells[3]) cells[3].innerHTML = chgTag(r.change_pct);
  if (cells[4]) cells[4].innerHTML = signalTag(r.signal);
  if (cells[6]) cells[6].innerHTML = rsiTag(r.rsi);

  // Refresh entry zone and stop loss on every tick
  if (cells[7]) {
    const zone = (r.entry_low != null && r.entry_high != null)
      ? `${fmt(r.entry_low, d)} – ${fmt(r.entry_high, d)}`
      : '—';
    cells[7].innerHTML = `<span style="font-size:11px;">${zone}</span>`;
  }
  if (cells[8]) {
    const sl = r.stop_loss != null
      ? `<span style="color:var(--red);font-size:11px;">${fmt(r.stop_loss, d)}</span>`
      : '—';
    cells[8].innerHTML = sl;
  }

  // Redraw sparkline
  const canvas = row.querySelector('.spark');
  if (canvas && priceHistory[name] && priceHistory[name].length >= 2) {
    drawSparkline(canvas, priceHistory[name]);
  }
}

// ── Socket.io connection ───────────────────────────────────────────────
const socket = io({ transports: ['websocket', 'polling'] });

socket.on('connect', () => {
  document.getElementById('dot').className      = 'dot live';
  document.getElementById('ws-label').textContent  = 'LIVE';
  document.getElementById('ws-label').style.color  = '#00ff9d';
});

socket.on('disconnect', () => {
  document.getElementById('dot').className      = 'dot dead';
  document.getElementById('ws-label').textContent  = 'DISCONNECTED';
  document.getElementById('ws-label').style.color  = '#ff3d71';
});

socket.on('price_update', (data) => {
  tickCount++;
  document.getElementById('tick-ctr').textContent = `TICKS: ${tickCount}`;
  document.getElementById('ts').textContent        = data.last_updated;

  if (!tableBuilt || !document.getElementById(rowId('Bitcoin'))) {
    buildTable(data.signals);
  } else {
    data.signals.forEach(r => updateRow(r));
  }
});
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
#  FLASK + SOCKET.IO APP
# ─────────────────────────────────────────────
app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=os.getenv("CORS_ORIGINS", "*"),
                   async_mode="threading")

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route("/health")
def health():
    with _cache_lock:
        return jsonify({"status": "ok", "last_updated": _last_updated}), 200

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Initialising database...")
    init_db()

    log.info("Running first signal cycle (blocking)...")
    run_all_signals()   # populate cache before clients connect

    log.info("Starting background signal thread...")
    t_signals = threading.Thread(target=background_loop, daemon=True)
    t_signals.start()

    log.info("Starting WebSocket emission thread...")
    t_emit = threading.Thread(target=emission_loop, daemon=True)
    t_emit.start()

    log.info("Flask-SocketIO server starting on 0.0.0.0:8080")
    # allow_unsafe_werkzeug is acceptable here; use Gunicorn+gevent in production
    socketio.run(app, host="0.0.0.0", port=8080, debug=False,
                 allow_unsafe_werkzeug=True)
