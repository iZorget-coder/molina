#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timezone

# Map daily CSV -> (Yahoo ticker, hourly CSV)
TICKER_MAP = {
    "SP500_1d_data.csv":  ("^GSPC", "SP500_index_H1.csv"),
    "NASDAQ_1d_data.csv": ("^IXIC", "NASDAQ100_index_H1.csv"),
    "DOW_1d_data.csv":    ("^DJI",  "DOW_index_H1.csv"),
    "XAUUSD_1d_data.csv": ("GC=F",  "XAUUSD_H1.csv"),
}


def today_utc():
    """Today's date in UTC (GitHub Actions runs in UTC). Any bar dated today or later is still in progress."""
    return datetime.now(timezone.utc).date()


def get_yahoo_daily_bars(ticker, after_date, today):
    """Completed daily bars from Yahoo with after_date < Date < today, oldest first."""
    try:
        # A small window is enough to catch up after a weekend, a holiday or a missed run
        df = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            return []

        # yfinance may return a MultiIndex column; flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        bars = []
        for ts, row in df.iterrows():
            bar_date = ts.date()

            if bar_date >= today:
                continue  # today's bar is still forming
            if bar_date <= after_date:
                continue  # already in the CSV

            # Reject Yahoo's "corrupt row" case (NaN OHLC)
            if any(pd.isna(row[c]) for c in ("Open", "High", "Low", "Close")):
                print(f"  Yahoo returned a NaN/incomplete bar for {bar_date}")
                continue

            bars.append({
                "Date":   str(bar_date),
                "Open":   float(row["Open"]),
                "High":   float(row["High"]),
                "Low":    float(row["Low"]),
                "Close":  float(row["Close"]),
                "Volume": float(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
            })
        return bars
    except Exception as e:
        print(f"  Yahoo fetch failed: {e}")
        return []


def aggregate_hourly_bars(hourly_path, after_date, today):
    """Daily bars built from the hourly CSV for after_date < UTC date < today, oldest first."""
    if not os.path.exists(hourly_path):
        return []
    try:
        hourly = pd.read_csv(hourly_path)
        if hourly.empty:
            return []

        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
        hourly["date"] = hourly["timestamp"].dt.date
        hourly = hourly[(hourly["date"] > after_date) & (hourly["date"] < today)]
        if hourly.empty:
            return []

        bars = []
        for bar_date, day in hourly.sort_values("timestamp").groupby("date"):
            has_volume = "volume" in day and day["volume"].notna().any()
            bars.append({
                "Date":   str(bar_date),
                "Open":   float(day["open"].iloc[0]),
                "High":   float(day["high"].max()),
                "Low":    float(day["low"].min()),
                "Close":  float(day["close"].iloc[-1]),
                "Volume": float(day["volume"].sum()) if has_volume else 0,
            })
        return bars
    except Exception as e:
        print(f"  Hourly aggregation failed: {e}")
        return []


def update_daily_csv(daily_path, yahoo_ticker, hourly_path):
    if not os.path.exists(daily_path):
        print(f"Skipping {daily_path} (file not found)")
        return

    today = today_utc()
    daily = pd.read_csv(daily_path)
    changed = False

    # 0. Drop any row dated today or later (an unfinished candle added by an earlier run)
    row_dates = pd.to_datetime(daily["Date"]).dt.date
    keep = row_dates < today
    if not keep.all():
        dropped = ", ".join(str(d) for d in row_dates[~keep])
        print(f"  Removing in-progress row(s): {dropped}")
        daily = daily[keep].reset_index(drop=True)
        changed = True

    last_daily_date = pd.to_datetime(daily["Date"].iloc[-1]).date()
    print(f"{daily_path}: last date = {last_daily_date} (today UTC = {today}, newest allowed = {today - pd.Timedelta(days=1)})")

    # 1. Try Yahoo first
    bars = get_yahoo_daily_bars(yahoo_ticker, last_daily_date, today)
    source = "Yahoo"

    # 2. Fall back to hourly aggregation if Yahoo had no new completed bar
    if not bars:
        print("  Yahoo didn't have a new complete bar. Trying hourly fallback...")
        bars = aggregate_hourly_bars(hourly_path, last_daily_date, today)
        source = "Hourly"

    # 3. Append every completed day we're missing (never today)
    if bars:
        rows = []
        for latest in bars:
            row = {col: latest.get(col, None) for col in daily.columns}
            if "Adj Close" in daily.columns:
                row["Adj Close"] = row["Close"]
            rows.append(row)
            print(f"  ✅ Appended {latest['Date']} from {source} → Close: {row['Close']:.2f}")
        daily = pd.concat([daily, pd.DataFrame(rows)], ignore_index=True)
        changed = True
    else:
        print(f"  No new completed days to add. CSV is up to date through {last_daily_date}.")

    if changed:
        daily.to_csv(daily_path, index=False)


if __name__ == "__main__":
    for daily_file, (yahoo_ticker, hourly_file) in TICKER_MAP.items():
        update_daily_csv(daily_file, yahoo_ticker, hourly_file)

