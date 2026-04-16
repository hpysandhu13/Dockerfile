"""
╔══════════════════════════════════════════════════════════════╗
║     INSTITUTIONAL SIGNAL INTELLIGENCE BOT v5.0              ║
║     Smart Money Confluence Engine — WebSocket Live Feed      ║
║     Strategy: Sniper Calculation — FVG/OB + RSI Reset +      ║
║               EMA Pullback + RVOL ≥ 1.8 Institutional Gate  ║
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

# Strategy params — Intraday Confluence (EMA 9/50 + RSI Cross-50 + VSA)
EMA_FAST         = 9
EMA_SLOW         = 50
RSI_PERIOD       = 14
RSI_MID          = 50    # crossover trigger threshold
VOLUME_LOOKBACK  = 10
VOLUME_THRESHOLD = 1.50   # 50 % above avg
ATR_MULTIPLIER   = 1.0    # fallback stop (1× ATR) when no swing stop is available
MIN_CANDLES      = 70     # need at least EMA_SLOW + buffer

# Order Block / Sniper Entry params
OB_VOLUME_THRESHOLD     = 2.0    # >200 % of rolling avg — required for a valid OB
OB_VOLUME_LOOKBACK      = 20     # periods used to compute the OB volume average
OB_SEARCH_CANDLES       = 20     # how many candles back to look for an OB before the displacement
RSI_ZONE_LOW            = 48     # RSI must be inside [48, 52] for STRONG signal — Momentum Reset component
RSI_ZONE_HIGH           = 52
DISPLACEMENT_BARS       = 10     # swing high/low lookback to identify a displacement move
CRYPTO_MAX_ZONE_PCT     = 0.001  # max OB zone width: 0.1 % of price (crypto)
FOREX_MAX_ZONE_PIPS     = 5      # max OB zone width: 5 pips (forex / commodity)
RVOL_THRESHOLD          = 1.8    # Minimum Relative Volume to confirm institutional activity
ATR_DISPLACEMENT_PERIOD = 10     # ATR period used to qualify a displacement candle
ATR_DISPLACEMENT_MULT   = 2.5    # Body size must exceed this multiple of ATR(10)
MIN_FVG_LOOKBACK        = 2      # candles before displacement required to form a 3-candle FVG sequence

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
                    ema9        NUMERIC,
                    ema50       NUMERIC,
                    vsa         BOOLEAN
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
                  (asset, asset_class, signal, price, entry_low, entry_high, stop_loss, rsi, ema9, ema50, vsa)
                VALUES (%(asset)s, %(asset_class)s, %(signal)s, %(price)s,
                        %(entry_low)s, %(entry_high)s, %(stop_loss)s, %(rsi)s, %(ema9)s, %(ema50)s, %(vsa)s)
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

def rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    """Return the full RSI series (needed for crossover detection)."""
    delta  = series.diff().dropna()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l  = loss.ewm(com=period - 1, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def rsi(series: pd.Series, period: int = 14) -> float:
    return round(float(rsi_series(series, period).iloc[-1]), 2)

def atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(com=period - 1, adjust=False).mean().iloc[-1])

def institutional_loading(df: pd.DataFrame) -> bool:
    """True if latest volume ≥ 150% of 10-candle avg AND candle spread > 10-period avg spread (VSA)."""
    if len(df) < VOLUME_LOOKBACK + 1:
        return False
    recent_avg_vol = df["Volume"].iloc[-(VOLUME_LOOKBACK + 1):-1].mean()
    if recent_avg_vol == 0:
        return False
    if float(df["Volume"].iloc[-1]) < recent_avg_vol * VOLUME_THRESHOLD:
        return False
    spread     = df["High"] - df["Low"]
    avg_spread = float(spread.iloc[-(VOLUME_LOOKBACK + 1):-1].mean())
    return avg_spread > 0 and float(spread.iloc[-1]) > avg_spread


def rvol(df: pd.DataFrame, lookback: int = 20) -> float:
    """Return Relative Volume = current volume / SMA(volume, lookback).

    Used as an institutional-activity gate: RVOL < RVOL_THRESHOLD means
    there is insufficient participation to confirm a displacement move.
    """
    if len(df) < lookback + 1:
        return 0.0
    sma_vol = float(df["Volume"].iloc[-(lookback + 1):-1].mean())
    if sma_vol <= 0:
        return 0.0
    return float(df["Volume"].iloc[-1]) / sma_vol

# ─────────────────────────────────────────────
#  ORDER BLOCK / SNIPER ENTRY HELPERS
# ─────────────────────────────────────────────
def _has_fvg_near_zone(df: pd.DataFrame, zone_low: float, zone_high: float,
                       side: str) -> bool:
    """
    Return True if there is an unfilled Fair Value Gap in the last 30 candles
    that is adjacent to (or overlapping) the order-block zone.

    Bullish FVG : candle[i].low > candle[i-2].high  →  gap above candle[i-2]
    Bearish FVG : candle[i].high < candle[i-2].low  →  gap below candle[i-2]

    The zone must be within 3× its own width of the FVG edge to qualify.
    """
    highs  = df["High"].values
    lows   = df["Low"].values
    n      = len(df)
    zone_mid   = (zone_low + zone_high) / 2
    zone_width = zone_high - zone_low
    proximity  = zone_width * 3.0 if zone_width != 0 else zone_mid * 0.001

    for i in range(max(2, n - 30), n):
        # Bullish FVG
        if lows[i] > highs[i - 2]:
            fvg_low = highs[i - 2]
            if side == "bull" and (
                (zone_low <= fvg_low <= zone_high)
                or abs(zone_mid - fvg_low) <= proximity
            ):
                return True
        # Bearish FVG
        if highs[i] < lows[i - 2]:
            fvg_high = lows[i - 2]
            if side == "bear" and (
                (zone_low <= fvg_high <= zone_high)
                or abs(zone_mid - fvg_high) <= proximity
            ):
                return True

    return False


def find_order_block(
    df: pd.DataFrame, asset_class: str
) -> tuple:
    """
    Identify an institutional displacement and return a Sniper Entry Zone.

    Algorithm — Sniper Calculation
    --------------------------------
    1. Compute a rolling ATR(10) series for every candle.
    2. Scan backwards for a *Displacement Candle* whose body size
       (|Close − Open|) exceeds ATR_DISPLACEMENT_MULT × ATR(10).
    3. Fair Value Gap (FVG) — for the 3-candle sequence ending at the
       displacement candle (Candle 1 = disp−2, Candle 3 = displacement):
         • Bullish FVG : Low[C3] > High[C1]
         • Bearish FVG : High[C3] < Low[C1]
       Entry zone is set to the 0.5 Mean Threshold (midpoint) of the FVG.
    4. If no clean FVG exists, fall back to the Institutional Order Block —
       the last opposite-direction candle immediately before the displacement.
    5. RVOL gate : if RVOL < RVOL_THRESHOLD, return WAITING.
       Low RVOL signals insufficient institutional participation.
    6. OB candle volume gate : OB volume must exceed OB_VOLUME_THRESHOLD ×
       OB_VOLUME_LOOKBACK-period rolling average, otherwise return SCANNING.
    7. Narrow the entry zone to the asset-class maximum width constraint.
    8. When using an OB fallback zone, require an unfilled FVG adjacent to
       it (existing alignment guard).
    9. Stop Loss is the swing low (bull) or swing high (bear) of the prior
       DISPLACEMENT_BARS candles — the origin of the displacement move.

    Returns
    -------
    (entry_low, entry_high, status, ob_side, swing_stop)
        status     ∈ {'FOUND', 'SCANNING', 'WAITING'}
        ob_side    ∈ {'bull', 'bear', None}
        swing_stop : float | None — displacement-move stop level
    """
    min_required = OB_VOLUME_LOOKBACK + DISPLACEMENT_BARS + 5
    if len(df) < min_required:
        return None, None, "WAITING", None, None

    closes = df["Close"].values
    opens  = df["Open"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    vols   = df["Volume"].values
    n      = len(df)

    # ── ATR(10) series for displacement body-size gate ────────────────────
    prev_closes = np.empty_like(closes)
    prev_closes[0]  = closes[0]
    prev_closes[1:] = closes[:-1]
    tr_arr = np.maximum.reduce([
        highs - lows,
        np.abs(highs - prev_closes),
        np.abs(lows  - prev_closes),
    ])
    atr10_vals = pd.Series(tr_arr).ewm(
        com=ATR_DISPLACEMENT_PERIOD - 1, adjust=False
    ).mean().values
    body_sizes = np.abs(closes - opens)

    ob_low     = ob_high = None
    ob_side    = None
    ob_index   = None          # OB candle index (for volume gate)
    disp_index = None          # displacement candle index
    has_fvg    = False
    swing_stop = None

    search_start = n - 1
    search_end   = max(DISPLACEMENT_BARS + OB_VOLUME_LOOKBACK, n - 60)

    for i in range(search_start, search_end, -1):
        # ── Displacement candle check: body > 2.5× ATR(10) ───────────────
        if body_sizes[i] <= ATR_DISPLACEMENT_MULT * atr10_vals[i]:
            continue

        is_bull_disp = closes[i] > opens[i]
        swing_start  = max(0, i - DISPLACEMENT_BARS)

        # ── Try FVG first (requires at least MIN_FVG_LOOKBACK candles before displacement)
        if i >= MIN_FVG_LOOKBACK:
            if is_bull_disp and lows[i] > highs[i - 2]:
                # Bullish FVG: gap between High[C1] and Low[C3]
                fvg_mid  = (float(highs[i - 2]) + float(lows[i])) / 2.0
                ob_low   = fvg_mid
                ob_high  = fvg_mid
                ob_side  = "bull"
                has_fvg  = True
                disp_index = i
                swing_stop = float(lows[swing_start:i].min())
                # OB = last bearish candle before displacement (for volume gate)
                for j in range(i - 1, max(0, i - OB_SEARCH_CANDLES), -1):
                    if closes[j] < opens[j]:
                        ob_index = j
                        break
                break

            elif not is_bull_disp and highs[i] < lows[i - 2]:
                # Bearish FVG: gap between Low[C1] and High[C3]
                fvg_mid  = (float(lows[i - 2]) + float(highs[i])) / 2.0
                ob_low   = fvg_mid
                ob_high  = fvg_mid
                ob_side  = "bear"
                has_fvg  = True
                disp_index = i
                swing_stop = float(highs[swing_start:i].max())
                # OB = last bullish candle before displacement (for volume gate)
                for j in range(i - 1, max(0, i - OB_SEARCH_CANDLES), -1):
                    if closes[j] > opens[j]:
                        ob_index = j
                        break
                break

        # ── Fallback: Institutional Order Block (no clean FVG) ───────────
        if is_bull_disp:
            for j in range(i - 1, max(0, i - OB_SEARCH_CANDLES), -1):
                if closes[j] < opens[j]:
                    ob_index   = j
                    ob_side    = "bull"
                    ob_low     = float(lows[j])
                    ob_high    = float(highs[j])
                    disp_index = i
                    swing_stop = float(lows[swing_start:i].min())
                    break
        else:
            for j in range(i - 1, max(0, i - OB_SEARCH_CANDLES), -1):
                if closes[j] > opens[j]:
                    ob_index   = j
                    ob_side    = "bear"
                    ob_low     = float(lows[j])
                    ob_high    = float(highs[j])
                    disp_index = i
                    swing_stop = float(highs[swing_start:i].max())
                    break

        if ob_index is not None:
            break

    if ob_index is None and not has_fvg:
        return None, None, "WAITING", None, None

    # ── RVOL gate: require institutional participation ────────────────────
    rvol_val = rvol(df, lookback=OB_VOLUME_LOOKBACK)
    if rvol_val < RVOL_THRESHOLD:
        return None, None, "WAITING", None, None

    # ── OB candle volume gate ─────────────────────────────────────────────
    if ob_index is not None:
        vol_slice = vols[max(0, ob_index - OB_VOLUME_LOOKBACK):ob_index]
        avg_vol   = float(vol_slice.mean()) if len(vol_slice) > 0 else 0.0
        if avg_vol <= 0 or float(vols[ob_index]) < OB_VOLUME_THRESHOLD * avg_vol:
            return None, None, "SCANNING", None, None

    # ── Narrow zone to asset-class max width ──────────────────────────────
    price = float(closes[-1])
    mid   = (ob_low + ob_high) / 2.0

    if asset_class == "CRYPTO":
        max_width = price * CRYPTO_MAX_ZONE_PCT
    else:
        pip_size  = 0.01 if price > 20 else 0.0001
        max_width = FOREX_MAX_ZONE_PIPS * pip_size

    half    = max_width / 2.0
    ob_low  = mid - half
    ob_high = mid + half

    # ── FVG alignment guard (only for OB fallback zones) ─────────────────
    if not has_fvg and not _has_fvg_near_zone(df, ob_low, ob_high, ob_side):
        return None, None, "WAITING", None, None

    return ob_low, ob_high, "FOUND", ob_side, swing_stop


# ─────────────────────────────────────────────
#  SIGNAL LOGIC
# ─────────────────────────────────────────────
def compute_signal(df: pd.DataFrame, label: str, asset_class: str) -> dict:
    """
    Sniper Calculation signal using institutional displacement detection,
    FVG/OB entry zones, RSI Momentum Reset, EMA 9 pullback, and RVOL gate.

    Entry zone
    ----------
    * Derived from a Displacement Candle (body > 2.5× ATR(10)) followed by:
        - A Fair Value Gap: gap between High[C1] and Low[C3] in the
          3-candle displacement sequence.  Entry = midpoint (0.5 threshold).
        - Fallback: the Institutional Order Block (last opposite-direction
          candle before the displacement), provided OB volume > 200 % of the
          20-period rolling average and an unfilled FVG aligns with the zone.
    * RVOL (Current Volume / SMA Volume 20) must be ≥ 1.8 — otherwise the
      entry zone stays WAITING (insufficient institutional participation).
    * When no qualifying setup is found: entry_low / entry_high are None and
      entry_zone_label is 'WAITING'.
    * When displacement found but OB volume is insufficient: 'SCANNING...'.

    Signal grading — STRONG BUY / STRONG SELL
    -----------------------------------------
    All three conditions must hold simultaneously:
      1. Current price is within the Entry Zone (OB / FVG zone is FOUND).
      2. RSI(14) is between 48 and 52 (Momentum Reset band).
      3. EMA 9 is pulling back toward EMA 50 but has not yet crossed it
         (EMA 9 > EMA 50 for bull; EMA 9 declining toward EMA 50).

    Stop loss
    ---------
    * STRONG signal: swing low (bull) or swing high (bear) of the prior
      DISPLACEMENT_BARS candles — the origin of the displacement move.
    * Fallback / WEAK signal: ATR_MULTIPLIER × ATR(14) from current price.

    WEAK BUY / WEAK SELL (secondary)
    ---------------------------------
    At least 2 of 3 factors align: EMA trend, RSI cross-50, VSA.

    Never raises — all errors produce NEUTRAL.
    """
    base = {
        "asset":            label,
        "asset_class":      asset_class.upper(),
        "price":            None,
        "signal":           "NEUTRAL",
        "entry_low":        None,
        "entry_high":       None,
        "entry_zone_label": "WAITING",
        "entry_zone_locked": False,
        "stop_loss":        None,
        "stop_loss_locked": False,
        "rsi":              None,
        "ema9":             None,
        "ema50":            None,
        "vsa":              False,
        "rvol":             None,
        "error":            None,
    }

    try:
        if df is None or len(df) < MIN_CANDLES:
            base["error"] = "Insufficient data"
            return base

        df    = df.copy()
        close = df["Close"]
        price = float(close.iloc[-1])

        # ── Dual EMA trend detection ─────────────────────────────────────
        ema9_s    = ema(close, EMA_FAST)
        ema50_s   = ema(close, EMA_SLOW)
        ema9_val  = float(ema9_s.iloc[-1])
        ema50_val = float(ema50_s.iloc[-1])
        ema9_prev = float(ema9_s.iloc[-2]) if len(ema9_s) >= 2 else ema9_val

        # ── RSI cross-50 detection (needs two consecutive values) ────────
        rsi_s    = rsi_series(close, RSI_PERIOD)
        rsi_val  = round(float(rsi_s.iloc[-1]), 2)
        rsi_prev = float(rsi_s.iloc[-2]) if len(rsi_s) >= 2 else rsi_val

        atr_val   = atr(df)
        inst_load = institutional_loading(df)
        rvol_val  = rvol(df, lookback=OB_VOLUME_LOOKBACK)

        base["price"] = round(price, 5)
        base["rsi"]   = rsi_val
        base["ema9"]  = round(ema9_val, 5)
        base["ema50"] = round(ema50_val, 5)
        base["vsa"]   = inst_load
        base["rvol"]  = round(rvol_val, 2)

        sl_distance = ATR_MULTIPLIER * atr_val
        base["stop_loss"] = round(price - sl_distance, 5)

        # ── Order Block / FVG entry zone ─────────────────────────────────
        ob_low, ob_high, ob_status, ob_side, swing_stop = find_order_block(
            df, asset_class.upper()
        )

        if ob_status == "FOUND":
            base["entry_low"]         = round(ob_low, 5)
            base["entry_high"]        = round(ob_high, 5)
            base["entry_zone_label"]  = "FOUND"
            base["entry_zone_locked"] = True   # do not overwrite with realtime price
        elif ob_status == "SCANNING":
            base["entry_zone_label"] = "SCANNING..."
        else:
            base["entry_zone_label"] = "WAITING"

        # ── EMA trend helpers ────────────────────────────────────────────
        ema_bullish = price > ema9_val and ema9_val > ema50_val
        ema_bearish = price < ema9_val and ema9_val < ema50_val

        # ── RSI cross-50 trigger ─────────────────────────────────────────
        rsi_cross_above = rsi_prev < RSI_MID <= rsi_val
        rsi_cross_below = rsi_prev > RSI_MID >= rsi_val

        # ── STRONG signal conditions ─────────────────────────────────────
        # 1. Price is inside the entry zone
        # 2. RSI(14) between 48 and 52 — Momentum Reset
        # 3. EMA 9 pulling back toward EMA 50 but has not yet crossed it:
        #      bull → EMA9 > EMA50 AND EMA9 declining (ema9_val < ema9_prev)
        #      bear → EMA9 < EMA50 AND EMA9 rising   (ema9_val > ema9_prev)
        rsi_reset     = RSI_ZONE_LOW <= rsi_val <= RSI_ZONE_HIGH
        price_in_zone = ob_status == "FOUND" and ob_low <= price <= ob_high

        ema_bull_pullback = ema9_val > ema50_val and ema9_val < ema9_prev
        ema_bear_pullback = ema9_val < ema50_val and ema9_val > ema9_prev

        if price_in_zone and rsi_reset:
            if ob_side == "bull" and ema_bull_pullback:
                base["signal"] = "STRONG BUY ▲"
                if swing_stop is not None:
                    base["stop_loss"]        = round(swing_stop, 5)
                    base["stop_loss_locked"] = True
            elif ob_side == "bear" and ema_bear_pullback:
                base["signal"] = "STRONG SELL ▼"
                if swing_stop is not None:
                    base["stop_loss"]        = round(swing_stop, 5)
                    base["stop_loss_locked"] = True
                else:
                    base["stop_loss"] = round(price + sl_distance, 5)
        else:
            # ── WEAK fallback: existing 3-factor confluence ───────────────
            buy_score  = int(ema_bullish) + int(rsi_cross_above) + int(inst_load)
            sell_score = int(ema_bearish) + int(rsi_cross_below) + int(inst_load)

            if buy_score == 2:
                base["signal"] = "WEAK BUY ▲"
            elif sell_score == 2:
                base["signal"]    = "WEAK SELL ▼"
                base["stop_loss"] = round(price + sl_distance, 5)

    except Exception as e:
        log.warning(f"Signal compute error [{label}]: {e}")
        base["error"] = str(e)

    return base

# ─────────────────────────────────────────────
#  DATA FETCHERS
# ─────────────────────────────────────────────
def _apply_realtime_price(result: dict, rt: float, atr_val: float) -> None:
    """Override price and recalculate stop around the live price in-place.

    The entry zone is NOT updated when it was set by an Order Block / FVG
    (entry_zone_locked == True) — the zone must remain stable.

    The stop loss is NOT recalculated when it was derived from the
    displacement swing high/low (stop_loss_locked == True) — a fixed
    structural level that should not shift with the live price.
    """
    sl_distance = ATR_MULTIPLIER * atr_val
    result["price"] = round(rt, 5)
    if not result.get("entry_zone_locked", False):
        # No OB/FVG zone: clear any stale price-based zone so it does not display
        result["entry_low"]  = None
        result["entry_high"] = None
    if not result.get("stop_loss_locked", False):
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
                    "ema9": None, "ema50": None, "vsa": False,
                    "entry_low": None, "entry_high": None,
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
                "ema9": None, "ema50": None, "vsa": False,
                "entry_low": None, "entry_high": None,
                "stop_loss": None, "error": str(e)}


def fetch_ccxt(symbol: str, label: str, exchange_id: str = "binance") -> dict:
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})

        # Fetch 200 candles (plenty for EMA50 + buffer)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1d", limit=200)
        if not ohlcv or len(ohlcv) < MIN_CANDLES:
            return {"asset": label, "asset_class": "CRYPTO",
                    "signal": "NEUTRAL", "price": None, "rsi": None,
                    "ema9": None, "ema50": None, "vsa": False,
                    "entry_low": None, "entry_high": None,
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
                "ema9": None, "ema50": None, "vsa": False,
                "entry_low": None, "entry_high": None,
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
                "ema9":        r.get("ema9"),
                "ema50":       r.get("ema50"),
                "vsa":         r.get("vsa", False),
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
    <div class="tagline">Sniper Calculation Engine · FVG/OB Entry · RSI Reset 48–52 · EMA Pullback · RVOL ≥ 1.8</div>
  </div>
  <div class="hdr-right">
    <div class="stat-item">
      <span class="stat-label">Strategy</span>
      <span class="stat-val" style="font-size:10px;">FVG/OB · RSI48-52 · RVOL≥1.8</span>
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
  <span class="badge">Displacement Body &gt; 2.5× ATR(10)</span>
  <span class="badge">FVG: High[C1]→Low[C3] Midpoint Entry</span>
  <span class="badge">RSI Reset 48–52</span>
  <span class="badge">EMA 9 Pullback → EMA 50</span>
  <span class="badge">RVOL ≥ 1.8 Institutional Gate</span>
  <span class="badge">Stop = Displacement Swing High/Low</span>
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
  const cls = v < 50 ? 'rsi-low' : v > 50 ? 'rsi-high' : '';
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

// ── Entry Zone display helper ──────────────────────────────────────────
function zoneDisplay(r) {
  const d = pDec(r.asset);
  const lbl = r.entry_zone_label || '';
  if (r.entry_low != null && r.entry_high != null) {
    // Valid OB zone: show the tight price spread
    return `<span style="color:var(--cyan);">${fmt(r.entry_low, d)}&nbsp;–&nbsp;${fmt(r.entry_high, d)}</span>`;
  }
  if (lbl === 'SCANNING...') {
    return `<span style="color:var(--amber);letter-spacing:1px;">SCANNING...</span>`;
  }
  // WAITING or any other state: empty cell
  return `<span style="color:var(--muted);">WAITING</span>`;
}

// ── Build all <td> content for a row ──────────────────────────────────
function buildCells(r, dir) {
  const d   = pDec(r.asset);
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
    <td style="font-size:11px;">${zoneDisplay(r)}</td>
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
    cells[7].innerHTML = `<span style="font-size:11px;">${zoneDisplay(r)}</span>`;
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
