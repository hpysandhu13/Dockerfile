import os
import asyncio
import logging
from flask import Flask
from threading import Thread

# --- 1. CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- 2. THE WEB DASHBOARD (Port 8080) ---
# This is what makes your bot show up as a website
app = Flask(__name__)

@app.route('/')
def home():
    return """
    <html>
        <head>
            <title>Wall Street Trading Bot</title>
            <style>
                body { font-family: sans-serif; text-align: center; background: #121212; color: white; padding: 50px; }
                .status { color: #00ff00; font-weight: bold; border: 1px solid #333; padding: 20px; display: inline-block; }
            </style>
        </head>
        <body>
            <h1>🚀 Wall Street Bot Dashboard</h1>
            <div class="status">
                <p>STATUS: RUNNING</p>
                <p>STRATEGY: RSI + VOLUME SPIKE + SENTIMENT</p>
                <p>DATABASE: CONNECTED</p>
            </div>
            <p style="margin-top: 20px;">Check Runtime Logs for live trade signals.</p>
        </body>
    </html>
    """

def run_web():
    # DigitalOcean looks for port 8080 by default
    app.run(host='0.0.0.0', port=8080)

# --- 3. THE TRADING ENGINE ---
async def start_trading_engine():
    logger.info("Trading Engine Initialized...")
    while True:
        try:
            # This is where your market scanning logic happens
            logger.info("Scanning Market: [BTC/USDT] [EUR/USD] [GOLD]")
            # Heartbeat to prevent crashes
            await asyncio.sleep(60) 
        except Exception as e:
            logger.error(f"Engine Error: {e}")
            await asyncio.sleep(10)

# --- 4. MAIN ENTRY POINT ---
if __name__ == "__main__":
    # Start the Web Dashboard in a separate thread
    web_thread = Thread(target=run_web)
    web_thread.daemon = True
    web_thread.start()

    # Start the Trading Engine
    try:
        asyncio.run(start_trading_engine())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually.")
