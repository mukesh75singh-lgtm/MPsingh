import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_scalar(value):
    """
    Extracts a scalar value from a pandas Series or a numpy array.
    If the value is already a scalar, it returns it as is.
    """
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.iloc[0]
    elif isinstance(value, np.ndarray):
        return value.item()
    else:
        return value

def check_stock(ticker):
    """
    Checks if a stock meets the screening criteria using yfinance.
    """
    try:
        # Append .NS for Indian stocks
        yf_ticker = ticker + ".NS"

        # Download historical data for the last 400 days
        hist_data = yf.download(yf_ticker, period="400d", progress=False)

        if hist_data.empty:
            logging.warning(f"No historical data for {ticker}")
            return False

        # Calculate 200 DMA
        hist_data['DMA200'] = hist_data['Close'].rolling(window=200).mean()

        # Get the last 200 DMA and last price
        last_dma = get_scalar(hist_data['DMA200'].iloc[-1])
        last_price = get_scalar(hist_data['Close'].iloc[-1])

        if pd.isna(last_dma):
            logging.info(f"{ticker}: Not enough data for 200 DMA.")
            return False

        if not ((last_price >= 0.95 * last_dma) and (last_price <= 1.05 * last_dma)):
            logging.info(f"{ticker}: Fails 200 DMA check. Last Price: {last_price:.2f}, 200 DMA: {last_dma:.2f}")
            return False
        logging.info(f"{ticker}: Passes 200 DMA check.")

        # 2. Volume Check
        last_6_days_volume = hist_data['Volume'].tail(6)
        if len(last_6_days_volume) < 6:
            logging.warning(f"Not enough volume data for {ticker}. Found {len(last_6_days_volume)} days.")
            return False

        latest_volume = get_scalar(last_6_days_volume.iloc[-1])
        avg_5_day_volume = get_scalar(last_6_days_volume.iloc[:-1].mean())

        if latest_volume < 3 * avg_5_day_volume:
            logging.info(f"{ticker}: Fails volume check. Latest: {latest_volume}, 5-day avg: {avg_5_day_volume:.2f}")
            return False

        logging.info(f"{ticker}: Passes volume check (using total traded volume as a proxy).")
        return True

    except Exception as e:
        logging.error(f"An error occurred while checking {ticker}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def main():
    """
    Main function to run the screener.
    """
    try:
        with open('june-25 (2).csv', 'r') as f:
            content = f.read()

        tickers = content.strip().replace('"', '').split(',')
        cleaned_tickers = [t.split(':')[1] for t in tickers if ':' in t]

        logging.info(f"Starting screener for {len(cleaned_tickers)} stocks...")

        logging.warning("--- IMPORTANT ---")
        logging.warning("This script uses total traded volume as a proxy for delivery volume.")
        logging.warning("The check for '3x delivery of 5 days average delivery' is based on this proxy.")
        logging.warning("-----------------")

        found_stocks = []
        for ticker in cleaned_tickers:
            logging.info(f"--- Checking {ticker} ---")
            if check_stock(ticker):
                found_stocks.append(ticker)
                print(f"\n*** Stock Found: {ticker} ***\n")

        logging.info(f"\n--- Screener Finished ---")
        logging.info(f"Found {len(found_stocks)} stocks: {', '.join(found_stocks)}")

    except FileNotFoundError:
        logging.error("Error: 'june-25 (2).csv' not found. Please make sure the file is in the same directory.")
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
