"""
╔══════════════════════════════════════════════════════════════╗
║     INSTITUTIONAL SIGNAL INTELLIGENCE BOT v2.0              ║
║     Smart Money Confluence Engine                            ║
║     Strategy: 200 EMA + RSI(14) + Volume Spread Analysis    ║
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
        {"id": "BTC/USDT",  "label": "Bitcoin",        "exchange": "binance"},
        {"id": "ETH/USDT",  "label": "Ethereum",       "exchange": "binance"},
    ],
    "forex": [
        {"id": "EURUSD=X",  "label": "EUR/USD"},
        {"id": "GBPUSD=X",  "label": "GBP/USD"},
    ],
    "commodity": [
        {"id": "GC=F",      "label": "Gold"},
        {"id": "SI=F",      "label": "Silver"},
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

REFRESH_SECONDS  = 60

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

        # ── STRONG BUY confluences ──────────────────────────────────────
        if price > ema200_val and rsi_val < RSI_OVERSOLD and vol_spike:
            base["signal"]    = "STRONG BUY ▲"
            base["entry_low"] = round(price * 0.999, 5)   # tight entry band
            base["entry_high"]= round(price * 1.001, 5)
            base["stop_loss"] = round(price - sl_distance, 5)

        # ── STRONG SELL confluences ─────────────────────────────────────
        elif price < ema200_val and rsi_val > RSI_OVERBOUGHT and vol_spike:
            base["signal"]    = "STRONG SELL ▼"
            base["entry_low"] = round(price * 0.999, 5)
            base["entry_high"]= round(price * 1.001, 5)
            base["stop_loss"] = round(price + sl_distance, 5)

    except Exception as e:
        log.warning(f"Signal compute error [{label}]: {e}")
        base["error"] = str(e)

    return base

# ─────────────────────────────────────────────
#  DATA FETCHERS
# ─────────────────────────────────────────────
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
        return compute_signal(raw, label, asset_class)

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

        return compute_signal(df, label, "CRYPTO")

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
        global _signal_cache, _last_updated
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

# ─────────────────────────────────────────────
#  HTML DASHBOARD
# ─────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Institutional Signal Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@700;800&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:        #07090f;
    --surface:   #0e1117;
    --border:    #1c2030;
    --accent:    #00e5ff;
    --buy:       #00ff9d;
    --sell:      #ff3d71;
    --neutral:   #4a5278;
    --text:      #cdd6f4;
    --muted:     #6272a4;
    --gold:      #f5c542;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    min-height: 100vh;
    padding: 2rem;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.2rem;
  }
  .logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #fff;
  }
  .logo span { color: var(--accent); }
  .meta {
    font-size: 0.7rem;
    color: var(--muted);
    text-align: right;
    line-height: 1.8;
  }
  .meta .ts { color: var(--accent); }
  .badge-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 1.8rem;
    flex-wrap: wrap;
  }
  .badge {
    font-size: 0.65rem;
    padding: 0.25rem 0.65rem;
    border: 1px solid var(--border);
    border-radius: 2px;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
  }
  thead tr {
    border-bottom: 1px solid var(--accent);
  }
  th {
    padding: 0.6rem 1rem;
    text-align: left;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
  }
  tbody tr {
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }
  tbody tr:hover { background: #ffffff06; }
  td {
    padding: 0.85rem 1rem;
    vertical-align: middle;
  }
  .asset-name { color: #fff; font-weight: 600; font-size: 0.82rem; }
  .asset-class {
    display: inline-block;
    font-size: 0.6rem;
    padding: 0.1rem 0.4rem;
    border-radius: 2px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
  }
  .cls-crypto    { background: #1a1440; color: #a78bfa; border: 1px solid #2d1f66; }
  .cls-forex     { background: #0d2035; color: #38bdf8; border: 1px solid #1a3d5c; }
  .cls-commodity { background: #1f1800; color: var(--gold); border: 1px solid #4a3800; }
  .price { color: #fff; font-weight: 600; font-size: 0.85rem; }
  .signal {
    font-weight: 600;
    font-size: 0.78rem;
    padding: 0.3rem 0.7rem;
    border-radius: 3px;
    display: inline-block;
    letter-spacing: 0.05em;
  }
  .sig-buy     { background: #00271a; color: var(--buy);  border: 1px solid #00ff9d44; }
  .sig-sell    { background: #2a0010; color: var(--sell); border: 1px solid #ff3d7144; }
  .sig-neutral { background: #111420; color: var(--neutral); border: 1px solid #1c2030; }
  .zone        { color: var(--text); font-size: 0.75rem; }
  .sl-val      { color: var(--sell); font-size: 0.75rem; }
  .rsi-chip {
    display: inline-block;
    font-size: 0.68rem;
    padding: 0.15rem 0.45rem;
    border-radius: 2px;
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .rsi-low    { color: var(--buy);  border-color: #00ff9d44; background: #00271a; }
  .rsi-high   { color: var(--sell); border-color: #ff3d7144; background: #2a0010; }
  .err { color: #ff6b6b; font-size: 0.7rem; }
  .spinner {
    display: inline-block;
    width: 8px; height: 8px;
    border: 1px solid var(--muted);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 6px;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  footer {
    margin-top: 2.5rem;
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.06em;
    border-top: 1px solid var(--border);
    padding-top: 1rem;
    display: flex;
    justify-content: space-between;
  }
  #status { color: var(--accent); }
  @media (max-width: 768px) {
    body { padding: 1rem; }
    table { font-size: 0.7rem; }
    th, td { padding: 0.6rem 0.5rem; }
  }
</style>
</head>
<body>
<header>
  <div>
    <div class="logo">SIGNAL<span>IQ</span></div>
    <div style="font-size:0.65rem;color:var(--muted);margin-top:0.3rem;">Institutional Smart Money Confluence Engine</div>
  </div>
  <div class="meta">
    Strategy: 200 EMA · RSI(14) · Volume Spread Analysis<br/>
    Last Updated: <span class="ts" id="ts">—</span>
  </div>
</header>

<div class="badge-row">
  <div class="badge">EMA 200 Trend Filter</div>
  <div class="badge">RSI &lt;30 / &gt;70 Extreme</div>
  <div class="badge">Vol Spike ≥ 150% Avg</div>
  <div class="badge">Stop = 1.5× ATR</div>
</div>

<table>
  <thead>
    <tr>
      <th>Asset</th>
      <th>Class</th>
      <th>Price</th>
      <th>Signal</th>
      <th>Entry Zone</th>
      <th>Stop Loss</th>
      <th>RSI(14)</th>
    </tr>
  </thead>
  <tbody id="tbody">
    <tr><td colspan="7" style="text-align:center;padding:3rem;color:var(--muted);">
      <span class="spinner"></span> Fetching institutional data...
    </td></tr>
  </tbody>
</table>

<footer>
  <span>SIGNALIQ · For informational purposes only · Not financial advice</span>
  <span id="status">Initialising...</span>
</footer>

<script>
  const fmt = (v, d=5) => v != null ? Number(v).toFixed(d) : '—';

  function classTag(cls) {
    const map = { CRYPTO:'crypto', FOREX:'forex', COMMODITY:'commodity' };
    const labels = { CRYPTO:'Crypto', FOREX:'Forex', COMMODITY:'Commodity' };
    const k = map[cls] || 'forex';
    return `<span class="asset-class cls-${k}">${labels[cls] || cls}</span>`;
  }

  function signalTag(sig) {
    if (!sig) return `<span class="signal sig-neutral">NEUTRAL</span>`;
    if (sig.includes('BUY'))  return `<span class="signal sig-buy">${sig}</span>`;
    if (sig.includes('SELL')) return `<span class="signal sig-sell">${sig}</span>`;
    return `<span class="signal sig-neutral">NEUTRAL</span>`;
  }

  function rsiTag(v) {
    if (v == null) return '—';
    const cls = v < 30 ? 'rsi-low' : v > 70 ? 'rsi-high' : '';
    return `<span class="rsi-chip ${cls}">${v}</span>`;
  }

  async function refresh() {
    document.getElementById('status').innerHTML = '<span class="spinner"></span>Refreshing...';
    try {
      const res  = await fetch('/api/signals');
      const data = await res.json();
      const tbody = document.getElementById('tbody');
      const pDec = (asset) => {
        if (['Bitcoin','Ethereum'].includes(asset)) return 2;
        if (['Gold','Silver'].includes(asset)) return 3;
        return 5;
      };
      tbody.innerHTML = data.signals.map(r => {
        const d = pDec(r.asset);
        const zone = (r.entry_low != null && r.entry_high != null)
          ? `${fmt(r.entry_low,d)} – ${fmt(r.entry_high,d)}`
          : '—';
        const sl   = r.stop_loss != null ? `<span class="sl-val">${fmt(r.stop_loss,d)}</span>` : '—';
        const err  = r.error ? `<br/><span class="err">⚠ ${r.error}</span>` : '';
        return `<tr>
          <td><span class="asset-name">${r.asset}</span>${err}</td>
          <td>${classTag(r.asset_class)}</td>
          <td><span class="price">${r.price != null ? fmt(r.price, d) : '—'}</span></td>
          <td>${signalTag(r.signal)}</td>
          <td class="zone">${zone}</td>
          <td>${sl}</td>
          <td>${rsiTag(r.rsi)}</td>
        </tr>`;
      }).join('');
      document.getElementById('ts').textContent = data.last_updated;
      document.getElementById('status').textContent = 'Live · Auto-refresh every 60s';
    } catch(e) {
      document.getElementById('status').textContent = '⚠ Fetch error — retrying...';
    }
  }

  refresh();
  setInterval(refresh, 60000);
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/signals")
def api_signals():
    with _cache_lock:
        return jsonify({
            "last_updated": _last_updated,
            "signals":      _signal_cache,
        })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "last_updated": _last_updated}), 200

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Initialising database...")
    init_db()

    log.info("Running first signal cycle (blocking)...")
    run_all_signals()   # run once synchronously so dashboard loads with data

    log.info("Starting background signal thread...")
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()

    log.info("Flask server starting on 0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
