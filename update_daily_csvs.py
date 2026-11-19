import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Map daily CSV -> (Yahoo ticker, hourly CSV)
TICKER_MAP = {
    "SP500_1d_data.csv":  ("^GSPC", "SP500_index_H1.csv"),
    "NASDAQ_1d_data.csv": ("^IXIC", "NASDAQ100_index_H1.csv"),
    "DOW_1d_data.csv":    ("^DJI",  "DOW_index_H1.csv"),
    "XAUUSD_1d_data.csv": ("GC=F",  "XAUUSD_H1.csv"),
}

def get_latest_yahoo_daily(ticker):
    """Fetch the last 5 days from Yahoo, return the most recent complete bar or None."""
    try:
        # Fetch a small window to capture the latest bar
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            return None
        
        # yfinance may return a MultiIndex column if ticker is a list; flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        latest = df.iloc[-1]
        
        # Reject if OHLC are NaN (Yahoo's "corrupt row" case)
        if pd.isna(latest["Open"]) or pd.isna(latest["High"]) or pd.isna(latest["Low"]) or pd.isna(latest["Close"]):
            print(f"  Yahoo returned a NaN/incomplete bar for {latest.name.date()}")
            return None
            
        return {
            "Date":   str(latest.name.date()),
            "Open":   float(latest["Open"]),
            "High":   float(latest["High"]),
            "Low":    float(latest["Low"]),
            "Close":  float(latest["Close"]),
            "Volume": float(latest["Volume"]) if not pd.isna(latest["Volume"]) else 0,
        }
    except Exception as e:
        print(f"  Yahoo fetch failed: {e}")
        return None

def aggregate_latest_hourly(hourly_path):
    """Aggregate the most recent complete UTC date from the hourly CSV."""
    if not os.path.exists(hourly_path):
        return None
    try:
        hourly = pd.read_csv(hourly_path)
        if hourly.empty:
            return None
        
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
        hourly["date"] = hourly["timestamp"].dt.date
        latest_date = hourly["date"].max()
        day = hourly[hourly["date"] == latest_date].sort_values("timestamp")
        
        if day.empty:
            return None
            
        return {
            "Date":   str(latest_date),
            "Open":   float(day["open"].iloc[0]),
            "High":   float(day["high"].max()),
            "Low":    float(day["low"].min()),
            "Close":  float(day["close"].iloc[-1]),
            "Volume": float(day["volume"].sum()) if "volume" in day and day["volume"].notna().any() else 0,
        }
    except Exception as e:
        print(f"  Hourly aggregation failed: {e}")
        return None

def update_daily_csv(daily_path, yahoo_ticker, hourly_path):
    if not os.path.exists(daily_path):
        print(f"Skipping {daily_path} (file not found)")
        return

    daily = pd.read_csv(daily_path)
    last_daily_date = str(pd.to_datetime(daily["Date"].iloc[-1]).date())
    
    print(f"{daily_path}: last date = {last_daily_date}")

    # 1. Try Yahoo first
    latest = get_latest_yahoo_daily(yahoo_ticker)
    source = "Yahoo"

    # 2. Fallback to hourly aggregation if Yahoo failed or returned an old date
    if latest is None or latest["Date"] <= last_daily_date:
        print(f"  Yahoo didn't have a new complete bar. Trying hourly fallback...")
        latest = aggregate_latest_hourly(hourly_path)
        source = "Hourly"

    if latest is None:
        print(f"  No new data available from either source.")
        return

    # 3. Final check: only append if it's actually a new date
    if latest["Date"] <= last_daily_date:
        print(f"  Latest available date ({latest['Date']}) is not newer than CSV ({last_daily_date}). Skipping.")
        return

    # 4. Append
    row = {col: latest.get(col, None) for col in daily.columns}
    if "Adj Close" in daily.columns:
        row["Adj Close"] = row["Close"]

    daily = pd.concat([daily, pd.DataFrame([row])], ignore_index=True)
    daily.to_csv(daily_path, index=False)
    print(f"  ✅ Appended {latest['Date']} from {source} → Close: {row['Close']:.2f}")

if __name__ == "__main__":
    for daily_file, (yahoo_ticker, hourly_file) in TICKER_MAP.items():
        update_daily_csv(daily_file, yahoo_ticker, hourly_file)