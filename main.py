
import time
"""
Advanced Intraday Trading Bot
- Crypto via ccxt
- Forex / Commodities via yfinance
- High-Probability signals: RSI + Volume Spike + Sentiment
- Risk Management: 1% max risk per trade + Trailing Stop-Loss
- Signal logging to PostgreSQL (DATABASE_URL env var)
- Fully async for maximum throughput
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import asyncpg
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
import yfinance as yf
from textblob import TextBlob

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

TRADING_CYCLE_INTERVAL = 60  # seconds between each trading cycle
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# Crypto symbols traded via ccxt (exchange:symbol)
CRYPTO_SYMBOLS: list[str] = os.getenv(
    "CRYPTO_SYMBOLS", "BTC/USDT,ETH/USDT"
).split(",")

# Forex / Commodity tickers via yfinance (e.g. EURUSD=X, GC=F)
FOREX_SYMBOLS: list[str] = os.getenv(
    "FOREX_SYMBOLS", "EURUSD=X,GC=F"
).split(",")

# Exchange name for ccxt (default: binance)
EXCHANGE_NAME: str = os.getenv("EXCHANGE_NAME", "binance")
EXCHANGE_API_KEY: str = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET: str = os.getenv("EXCHANGE_API_SECRET", "")

# NewsAPI key (optional – enables live headlines for sentiment)
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

# How often (seconds) to run one full scan of all symbols
TRADING_CYCLE_INTERVAL: int = int(os.getenv("TRADING_CYCLE_INTERVAL", "60"))

# Technical-analysis parameters
RSI_PERIOD: int = 14
RSI_OVERSOLD: float = 30.0
RSI_OVERBOUGHT: float = 70.0
VOLUME_SPIKE_MULTIPLIER: float = 1.5   # current vol > N * average vol

# Risk parameters
ACCOUNT_BALANCE: float = float(os.getenv("ACCOUNT_BALANCE", "10000"))
MAX_RISK_PER_TRADE_PCT: float = 0.01   # 1 %
TRAILING_STOP_PCT: float = float(os.getenv("TRAILING_STOP_PCT", "0.02"))  # 2 %

def run_trading_cycle():
    """Execute a single trading cycle: fetch data, evaluate signals, place orders."""
    logger.info("Running trading cycle...")
    # TODO: Add real market data fetching, signal logic, and order placement here.
# Sentiment threshold (TextBlob polarity: -1 to +1)
SENTIMENT_POSITIVE_THRESHOLD: float = 0.05

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

def main():
    logger.info("Trading bot started.")
    while True:
@dataclass
class Signal:
    symbol: str
    direction: str          # "BUY" | "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float    # units / contracts
    rsi: float
    volume_spike: bool
    sentiment_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    current_price: float
    stop_loss: float        # trailing – updated as price moves
    position_size: float


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

_db_pool: Optional[asyncpg.Pool] = None


async def get_db_pool() -> Optional[asyncpg.Pool]:
    global _db_pool
    if _db_pool is None and DATABASE_URL:
        try:
            run_trading_cycle()
            _db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
            await _ensure_table(_db_pool)
            logger.info("PostgreSQL pool created.")
        except Exception as exc:
            logger.error("Error during trading cycle: %s", exc)
        time.sleep(TRADING_CYCLE_INTERVAL)
            logger.error("Could not connect to PostgreSQL: %s", exc)
            _db_pool = None
    return _db_pool


async def _ensure_table(pool: asyncpg.Pool) -> None:
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS trading_signals (
            id              SERIAL PRIMARY KEY,
            symbol          TEXT        NOT NULL,
            direction       TEXT        NOT NULL,
            entry_price     NUMERIC     NOT NULL,
            stop_loss       NUMERIC     NOT NULL,
            take_profit     NUMERIC     NOT NULL,
            position_size   NUMERIC     NOT NULL,
            rsi             NUMERIC,
            volume_spike    BOOLEAN,
            sentiment_score NUMERIC,
            timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


async def log_signal(signal: Signal) -> None:
    pool = await get_db_pool()
    if pool is None:
        logger.warning("DB unavailable – signal not persisted: %s", signal)
        return
    try:
        await pool.execute(
            """
            INSERT INTO trading_signals
                (symbol, direction, entry_price, stop_loss, take_profit,
                 position_size, rsi, volume_spike, sentiment_score, timestamp)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            """,
            signal.symbol,
            signal.direction,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
            signal.position_size,
            signal.rsi,
            signal.volume_spike,
            signal.sentiment_score,
            signal.timestamp,
        )
    except Exception as exc:
        logger.error("Failed to log signal to DB: %s", exc)


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------

async def fetch_crypto_ohlcv(
    exchange: ccxt_async.Exchange, symbol: str, timeframe: str = "5m", limit: int = 100
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV candles for a crypto symbol via ccxt."""
    try:
        raw = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df
    except Exception as exc:
        logger.error("ccxt fetch error for %s: %s", symbol, exc)
        return None


def fetch_forex_ohlcv(symbol: str, period: str = "1d", interval: str = "5m") -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for a forex / commodity ticker via yfinance (sync, run in executor)."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index, utc=True)
        # The index name differs by yfinance version / data type; normalise it.
        df.index.name = "timestamp"
        return df.reset_index()
    except Exception as exc:
        logger.error("yfinance fetch error for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Technical analysis
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> float:
    """Return the most recent RSI value."""
    if len(close) < period + 1:
        return 50.0  # neutral fallback
    delta = close.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def detect_volume_spike(volume: pd.Series, multiplier: float = VOLUME_SPIKE_MULTIPLIER) -> bool:
    """Return True when the latest candle's volume exceeds N × the rolling average."""
    if len(volume) < 2:
        return False
    avg_vol = volume.iloc[:-1].mean()
    if avg_vol == 0:
        return False
    return bool(volume.iloc[-1] > multiplier * avg_vol)


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

async def fetch_sentiment(symbol: str) -> float:
    """
    Return a sentiment polarity score in [-1, +1].

    If NEWS_API_KEY is set the function fetches live headlines from NewsAPI;
    otherwise it falls back to a TextBlob analysis of a static placeholder
    headline so the pipeline always produces a numeric score.
    """
    headlines: list[str] = []

    if NEWS_API_KEY:
        import aiohttp  # imported here to avoid hard dependency when unused

        base_symbol = symbol.split("/")[0].split("=")[0]
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={base_symbol}&sortBy=publishedAt&pageSize=5"
            f"&apiKey={NEWS_API_KEY}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    headlines = [
                        a.get("title", "") + " " + a.get("description", "")
                        for a in data.get("articles", [])
                    ]
        except Exception as exc:
            logger.warning("NewsAPI request failed for %s: %s", symbol, exc)

    if not headlines:
        # Placeholder headline when no live data is available
        headlines = [f"{symbol} market conditions are stable and moderately positive"]

    combined_text = " ".join(headlines)
    polarity: float = TextBlob(combined_text).sentiment.polarity
    return polarity


# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------

def calculate_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    max_risk_pct: float = MAX_RISK_PER_TRADE_PCT,
) -> float:
    """
    Kelly-inspired 1 % risk sizing.
    position_size = (account_balance * max_risk_pct) / |entry_price - stop_loss|
    """
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit == 0:
        return 0.0
    dollar_risk = account_balance * max_risk_pct
    return dollar_risk / risk_per_unit


def calculate_stop_loss(price: float, direction: str, pct: float = TRAILING_STOP_PCT) -> float:
    """Initial stop-loss distance from entry."""
    return price * (1 - pct) if direction == "BUY" else price * (1 + pct)


def calculate_take_profit(price: float, direction: str, reward_ratio: float = 2.0) -> float:
    """Take-profit at 2× the initial stop distance (2:1 R:R)."""
    stop_dist = price * TRAILING_STOP_PCT
    return price + reward_ratio * stop_dist if direction == "BUY" else price - reward_ratio * stop_dist


def update_trailing_stop(position: Position, pct: float = TRAILING_STOP_PCT) -> Position:
    """Ratchet the stop-loss upward (BUY) or downward (SELL) as price improves."""
    if position.direction == "BUY":
        new_stop = position.current_price * (1 - pct)
        if new_stop > position.stop_loss:
            position.stop_loss = new_stop
    else:
        new_stop = position.current_price * (1 + pct)
        if new_stop < position.stop_loss:
            position.stop_loss = new_stop
    return position


def is_stopped_out(position: Position) -> bool:
    """Return True when the current price has crossed the trailing stop."""
    if position.direction == "BUY":
        return position.current_price <= position.stop_loss
    return position.current_price >= position.stop_loss


# ---------------------------------------------------------------------------
# High-probability signal generator
# ---------------------------------------------------------------------------

async def evaluate_symbol(
    symbol: str,
    df: pd.DataFrame,
    source: str,  # "crypto" | "forex"
) -> Optional[Signal]:
    """
    Emit a signal only when ALL three conditions align:
      1. RSI is oversold (< RSI_OVERSOLD) → BUY  or  overbought (> RSI_OVERBOUGHT) → SELL
      2. Volume spike detected
      3. Sentiment score exceeds SENTIMENT_POSITIVE_THRESHOLD (BUY) or is negative (SELL)
    """
    if df is None or len(df) < RSI_PERIOD + 5:
        return None

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    current_price = float(close.iloc[-1])

    rsi = compute_rsi(close)
    vol_spike = detect_volume_spike(volume)
    sentiment = await fetch_sentiment(symbol)

    logger.info(
        "[%s] %s | price=%.5f RSI=%.1f vol_spike=%s sentiment=%.3f",
        source.upper(),
        symbol,
        current_price,
        rsi,
        vol_spike,
        sentiment,
    )

    direction: Optional[str] = None
    if rsi < RSI_OVERSOLD and vol_spike and sentiment > SENTIMENT_POSITIVE_THRESHOLD:
        direction = "BUY"
    elif rsi > RSI_OVERBOUGHT and vol_spike and sentiment < -SENTIMENT_POSITIVE_THRESHOLD:
        direction = "SELL"

    if direction is None:
        return None

    stop_loss = calculate_stop_loss(current_price, direction)
    take_profit = calculate_take_profit(current_price, direction)
    position_size = calculate_position_size(ACCOUNT_BALANCE, current_price, stop_loss)

    if position_size <= 0:
        return None

    signal = Signal(
        symbol=symbol,
        direction=direction,
        entry_price=current_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size,
        rsi=rsi,
        volume_spike=vol_spike,
        sentiment_score=sentiment,
    )
    logger.info("*** HIGH-PROBABILITY SIGNAL: %s ***", signal)
    return signal


# ---------------------------------------------------------------------------
# Trading cycle
# ---------------------------------------------------------------------------

async def _process_crypto_symbol(
    exchange: ccxt_async.Exchange, symbol: str
) -> None:
    df = await fetch_crypto_ohlcv(exchange, symbol)
    signal = await evaluate_symbol(symbol, df, "crypto")
    if signal:
        await log_signal(signal)


async def _process_forex_symbol(
    loop: asyncio.AbstractEventLoop, symbol: str
) -> None:
    df = await loop.run_in_executor(None, fetch_forex_ohlcv, symbol)
    signal = await evaluate_symbol(symbol, df, "forex")
    if signal:
        await log_signal(signal)


async def run_trading_cycle(exchange: ccxt_async.Exchange, loop: asyncio.AbstractEventLoop) -> None:
    """Scan all configured symbols once and log any generated signals."""
    tasks: list[asyncio.Task] = []

    # --- Crypto symbols ---
    for sym in CRYPTO_SYMBOLS:
        sym = sym.strip()
        if not sym:
            continue
        tasks.append(asyncio.create_task(_process_crypto_symbol(exchange, sym)))

    # --- Forex / Commodity symbols (yfinance is sync; run in thread pool) ---
    for sym in FOREX_SYMBOLS:
        sym = sym.strip()
        if not sym:
            continue
        tasks.append(asyncio.create_task(_process_forex_symbol(loop, sym)))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def main() -> None:
    logger.info("Trading bot starting up…")
    logger.info(
        "Crypto symbols: %s | Forex symbols: %s | Cycle: %ss",
        CRYPTO_SYMBOLS,
        FOREX_SYMBOLS,
        TRADING_CYCLE_INTERVAL,
    )

    # Initialise exchange
    exchange_cls = getattr(ccxt_async, EXCHANGE_NAME, None)
    if exchange_cls is None:
        raise ValueError(f"Unknown ccxt exchange: {EXCHANGE_NAME!r}")

    exchange: ccxt_async.Exchange = exchange_cls(
        {
            "apiKey": EXCHANGE_API_KEY,
            "secret": EXCHANGE_API_SECRET,
            "enableRateLimit": True,
        }
    )

    # Pre-warm DB connection
    await get_db_pool()

    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                await run_trading_cycle(exchange, loop)
            except Exception as exc:
                logger.error("Unhandled error in trading cycle: %s", exc, exc_info=True)
            await asyncio.sleep(TRADING_CYCLE_INTERVAL)
    finally:
        await exchange.close()
        pool = await get_db_pool()
        if pool:
            await pool.close()
        logger.info("Trading bot shut down cleanly.")


if __name__ == "__main__":
    main()
    asyncio.run(main())
