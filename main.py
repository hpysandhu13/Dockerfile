import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

TRADING_CYCLE_INTERVAL = 60  # seconds between each trading cycle


def run_trading_cycle():
    """Execute a single trading cycle: fetch data, evaluate signals, place orders."""
    logger.info("Running trading cycle...")
    # TODO: Add real market data fetching, signal logic, and order placement here.


def main():
    logger.info("Trading bot started.")
    while True:
        try:
            run_trading_cycle()
        except Exception as exc:
            logger.error("Error during trading cycle: %s", exc)
        time.sleep(TRADING_CYCLE_INTERVAL)


if __name__ == "__main__":
    main()
