#!/usr/bin/env python
# coding: utf-8

# In[34]:


import requests
import pandas as pd
from datetime import date, timedelta, datetime, timezone
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os
import asyncio

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# Fields: (dataset_path, ticker, outfile, tv_symbol, csv_symbol_tag)
# Futures symbols used so session length matches the 23-hour historical feed
INSTRUMENTS = {
    "sp500_idx":   ("indices/global",     "USA500.IDX",  "SP500_index_H1.csv",
                    "CME_MINI:ES1!",    "USA500"),
    "nasdaq_idx":  ("indices/global",     "USATECH.IDX", "NASDAQ100_index_H1.csv",
                    "CME_MINI:NQ1!",    "USATECH"),
    "dow_idx":     ("indices/global",     "USA30.IDX",   "DOW_index_H1.csv",
                    "CBOT_MINI:YM1!",   "USA30"),
    "gold":        ("commodities/global", "XAUUSD",      "XAUUSD_H1.csv",
                    "OANDA:XAUUSD",     "XAUUSD"),
}

TO_DOWNLOAD = ["sp500_idx", "nasdaq_idx", "dow_idx", "gold"]

START = date(2021, 11, 11)
END   = date.today()
CHUNK_DAYS = 180
BASE_URL   = "https://getmarkethistory.com/api/v1/bars"
MAX_WORKERS = 4

MIN_INTERVAL_SECONDS = 6.5
MAX_RETRIES          = 5


def _to_datetime_series(series):
    """Coerce any timestamp series (str or datetime) to UTC datetime."""
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")


def _ensure_datetime(df, col="timestamp"):
    """Idempotent: coerce the timestamp column to datetime if not already."""
    if df is None or df.empty or col not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = _to_datetime_series(df[col])
    return df


# ─────────────────────────────────────────────────────────────
# GLOBAL RATE LIMITER
# ─────────────────────────────────────────────────────────────

_rate_lock    = threading.Lock()
_last_request = 0.0

def rate_limited_get(url: str, params: dict, timeout: int = 120) -> requests.Response:
    global _last_request
    with _rate_lock:
        now = time.time()
        wait = MIN_INTERVAL_SECONDS - (now - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
    ticker = url.rstrip("/").split("/")[-2]
    print(f"  🌐 rate-limit slot → {ticker}  (t={time.time():.0f})", flush=True)
    return requests.get(url, params=params, timeout=timeout)


# ─────────────────────────────────────────────────────────────
# INCREMENTAL RESUME
# ─────────────────────────────────────────────────────────────

def last_saved_timestamp(csv_path: str):
    if not os.path.exists(csv_path):
        return None
    try:
        existing = pd.read_csv(csv_path, usecols=["timestamp"])
        if existing.empty:
            return None
        return _to_datetime_series(existing["timestamp"]).max()
    except Exception as e:
        print(f"  ⚠️  could not read {csv_path}: {e} — will re-download full history")
        return None


# ─────────────────────────────────────────────────────────────
# TRADINGVIEW FALLBACK (via tvkit)
# ─────────────────────────────────────────────────────────────

def fetch_from_tradingview(symbol: str, start: date, end: date,
                            csv_symbol_tag: str) -> pd.DataFrame:
    try:
        from tvkit.api.chart.ohlcv import OHLCV
    except ImportError:
        print(f"  ⚠️  tvkit not installed — pip install tvkit", flush=True)
        return pd.DataFrame()

    from_iso = start.isoformat()
    to_iso   = end.isoformat()
    print(f"  📺 TradingView fetch: {symbol}  1h  {from_iso} → {to_iso}", flush=True)

    result_box = {"bars": None, "error": None}

    async def _inner():
        async with OHLCV() as client:
            return await client.get_historical_ohlcv(
                exchange_symbol=symbol,
                interval="1H",
                start=from_iso,
                end=to_iso,
            )

    def _runner():
        try:
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                result_box["bars"] = new_loop.run_until_complete(_inner())
            finally:
                try:
                    new_loop.close()
                finally:
                    asyncio.set_event_loop(None)
        except Exception as e:
            result_box["error"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=120)

    if t.is_alive():
        print(f"  ❌ TradingView fetch timed out for {symbol}", flush=True)
        return pd.DataFrame()

    if result_box["error"] is not None:
        e = result_box["error"]
        print(f"  ❌ TradingView fetch failed ({symbol}): {e.__class__.__name__}: {e}",
              flush=True)
        return pd.DataFrame()

    bars = result_box["bars"]
    if not bars:
        print(f"  ℹ️  TradingView returned 0 bars for {symbol} "
              f"{from_iso} → {to_iso}", flush=True)
        return pd.DataFrame()

    rows = [{
        "timestamp": pd.to_datetime(b.timestamp, unit="s", utc=True),
        "open": float(b.open),
        "high": float(b.high),
        "low": float(b.low),
        "close": float(b.close),
        "volume": float(b.volume),
    } for b in bars]

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    df["symbol"]           = csv_symbol_tag
    df["symbol_id"]        = 0
    df["vwap"]             = None
    df["trade_count"]      = 0
    df["is_gap"]           = False
    df["gap_fill_method"]  = 0
    df["hours_since_prev"] = df["timestamp"].diff().dt.total_seconds() / 3600
    df["is_contiguous"]    = df["hours_since_prev"] == 1

    ordered = [
        "symbol", "timestamp", "symbol_id", "open", "high", "low", "close",
        "vwap", "volume", "trade_count", "is_gap", "gap_fill_method",
        "hours_since_prev", "is_contiguous",
    ]
    df = df[[c for c in ordered if c in df.columns]]

    print(f"  ✅ TradingView returned {len(df)} bars "
          f"({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})", flush=True)
    return df


# ─────────────────────────────────────────────────────────────
# DOWNLOAD ONE INSTRUMENT
# ─────────────────────────────────────────────────────────────

def download(dataset: str, ticker: str, start: date, end: date,
             outfile: str, tv_symbol: str = None, csv_symbol_tag: str = None,
             chunk_days: int = CHUNK_DAYS) -> pd.DataFrame:
    url = f"{BASE_URL}/{dataset}/{ticker}/1h"

    existing_df = None
    resume_from = start

    if os.path.exists(outfile):
        latest = last_saved_timestamp(outfile)
        if latest is not None:
            resume_from = max(start, latest.date())
            print(f"  📂 [{ticker}] found {outfile}, "
                  f"last bar = {latest} → resuming from {resume_from}",
                  flush=True)
            existing_df = pd.read_csv(outfile)
            existing_df = _ensure_datetime(existing_df)
        else:
            print(f"  📂 [{ticker}] {outfile} exists but is empty/unreadable "
                  f"→ full re-download", flush=True)
    else:
        print(f"  📂 [{ticker}] no {outfile} → full download from {start}",
              flush=True)

    if resume_from >= end:
        print(f"  ✅ [{ticker}] already up to date (last bar ≥ {end}), "
              f"skipping", flush=True)
        return existing_df

    # ── 1. download the missing range from getmarkethistory ──
    frames = []
    current = resume_from

    total_chunks = 0
    tmp = resume_from
    while tmp < end:
        tmp = min(tmp + timedelta(days=chunk_days), end)
        total_chunks += 1
    chunk_n = 0

    while current < end:
        chunk_n += 1
        chunk_end = min(current + timedelta(days=chunk_days), end)
        params = {
            "start":  current.isoformat(),
            "end":    chunk_end.isoformat(),
            "format": "csv",
            "limit":  5000,
        }

        print(f"  [{ticker}] chunk {chunk_n}/{total_chunks}  "
              f"{current} → {chunk_end}  (waiting for rate limit…)",
              flush=True)

        response = None
        for attempt in range(MAX_RETRIES):
            try:
                r = rate_limited_get(url, params)
            except requests.exceptions.RequestException as e:
                backoff = 15 * (attempt + 1)
                print(f"  🌐❌ [{ticker}] network error on {current}→{chunk_end}: "
                      f"{e.__class__.__name__} — retry in {backoff}s "
                      f"({attempt+1}/{MAX_RETRIES})", flush=True)
                time.sleep(backoff)
                continue
            if r.status_code == 429:
                backoff = 20 * (attempt + 1)
                print(f"  ⏳ [{ticker}] 429 on {current}→{chunk_end}, "
                      f"backing off {backoff}s (attempt {attempt+1}/{MAX_RETRIES})",
                      flush=True)
                time.sleep(backoff)
                continue
            r.raise_for_status()
            response = r
            break

        if response is None:
            print(f"  ❌ [{ticker}] giving up on {current}→{chunk_end} after retries",
                  flush=True)
            return existing_df if existing_df is not None else pd.DataFrame()

        page = pd.read_csv(StringIO(response.text))
        print(f"  ✅ [{ticker}] chunk {chunk_n}/{total_chunks}: "
              f"{len(page)} bars", flush=True)

        if len(page) >= 5000:
            print(f"  ⚠️  [{ticker}] hit 5000 cap in {current}→{chunk_end}",
                  flush=True)
        if len(page):
            frames.append(page)

        current = chunk_end

    new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    new_df = _ensure_datetime(new_df)

    # ── 2. determine true latest after getmarkethistory ──────
    if existing_df is not None and not existing_df.empty:
        _all = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        _all = new_df

    latest_ts = None
    if not _all.empty:
        _all = _ensure_datetime(_all)
        _all = _all.dropna(subset=["timestamp"])
        if not _all.empty:
            latest_ts = _all["timestamp"].max()

    latest_date = latest_ts.date() if latest_ts is not None else resume_from

    # ── 3. TradingView fallback for the gap ──────────────────
    today = date.today()
    if tv_symbol and latest_date < today:
        print(f"  📺 [{ticker}] getmarkethistory last={latest_date}, "
              f"today={today} → TradingView backfill "
              f"{latest_date} → {today}", flush=True)
        tv_df = fetch_from_tradingview(tv_symbol, latest_date, today,
                                        csv_symbol_tag=csv_symbol_tag or ticker.split(".")[0])
        tv_df = _ensure_datetime(tv_df)
        if not tv_df.empty:
            if latest_ts is not None:
                tv_df = tv_df[tv_df["timestamp"] > latest_ts]
            if not tv_df.empty:
                print(f"  ✅ [{ticker}] TradingView supplied "
                      f"{len(tv_df)} new bars", flush=True)
                new_df = pd.concat([new_df, tv_df], ignore_index=True)
            else:
                print(f"  ℹ️  [{ticker}] TradingView had no bars newer than "
                      f"{latest_ts}", flush=True)
        else:
            print(f"  ℹ️  [{ticker}] TradingView returned nothing", flush=True)

    # ── 4. merge ─────────────────────────────────────────────
    existing_df = _ensure_datetime(existing_df)
    new_df      = _ensure_datetime(new_df)

    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    if combined.empty:
        print(f"  ❌ [{ticker}] no data at all", flush=True)
        return pd.DataFrame()

    n_before = len(existing_df) if existing_df is not None and not existing_df.empty else 0
    combined = _ensure_datetime(combined)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    combined["hours_since_prev"] = combined["timestamp"].diff().dt.total_seconds() / 3600
    combined["is_contiguous"]    = combined["hours_since_prev"] == 1

    n_new = len(combined) - n_before
    if n_new != 0:
        print(f"  📈 [{ticker}] {n_new:+d} new bar(s) (total {len(combined)})", flush=True)
    else:
        print(f"  ℹ️  [{ticker}] no change (total {len(combined)})", flush=True)
    return combined


# ─────────────────────────────────────────────────────────────
# WRAPPER
# ─────────────────────────────────────────────────────────────

def fetch_one(key: str):
    dataset, ticker, outfile, tv_symbol, csv_symbol_tag = INSTRUMENTS[key]
    print(f"[start] {key} ({ticker})")
    df = download(dataset, ticker, START, END, outfile,
                   tv_symbol=tv_symbol, csv_symbol_tag=csv_symbol_tag)

    if df.empty:
        return key, ticker, outfile, df

    df = _ensure_datetime(df)
    df["hours_since_prev"] = df["timestamp"].diff().dt.total_seconds() / 3600
    df["is_contiguous"]    = df["hours_since_prev"] == 1

    df.to_csv(outfile, index=False)
    print(f"[done ] {key} ({ticker}): {len(df)} bars → {outfile}", flush=True)
    return key, ticker, outfile, df


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, k): k for k in TO_DOWNLOAD}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                k, ticker, outfile, df = fut.result()
                results[k] = df
            except Exception as e:
                print(f"[FAIL ] {key}: {e}")

    failed = [k for k in TO_DOWNLOAD if results.get(k) is None or results[k].empty]
    if failed:
        print(f"\n🔁 retrying failed instruments: {failed}", flush=True)
        for key in failed:
            try:
                k, ticker, outfile, df = fetch_one(key)
                results[k] = df
            except Exception as e:
                print(f"[FAIL ] {key} (2nd pass): {e}")

    print(f"\n========== SUMMARY ({time.time()-t0:.1f}s) ==========")
    for k in TO_DOWNLOAD:
        df = results.get(k)
        if df is None or df.empty:
            print(f"{k:12s} ❌ no data")
        else:
            print(f"{k:12s} ✅ {len(df):>6} bars | "
                  f"{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()

