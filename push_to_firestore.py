# push_to_firestore.py
import firebase_admin
from firebase_admin import credentials, firestore
import json
import os
import pandas as pd

# Load the service account from the environment variable
service_account_info = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred)

db = firestore.client()

# Load the predictions produced by market data.py
with open("predictions.json", "r") as f:
    predictions = json.load(f)

DAILY_FILES = {
    "SPX":    "SP500_1d_data.csv",
    "NDX":    "NASDAQ_1d_data.csv",
    "DJI":    "DOW_1d_data.csv",
    "XAUUSD": "XAUUSD_1d_data.csv",
}

HOURLY_FILES = {
    "SPX":    "SP500_index_H1.csv",
    "NDX":    "NASDAQ100_index_H1.csv",
    "DJI":    "DOW_index_H1.csv",
    "XAUUSD": "XAUUSD_H1.csv",
}

def daily_candles(symbol, n=100):
    path = DAILY_FILES.get(symbol)
    if not path or not os.path.exists(path):
        print(f"[warn] No daily CSV for {symbol} ({path})")
        return []
    df = pd.read_csv(path).tail(n)
    out = []
    for _, row in df.iterrows():
        try:
            out.append({
                "date":  str(row["Date"]),
                "open":  float(row["Open"]),
                "high":  float(row["High"]),
                "low":   float(row["Low"]),
                "close": float(row["Close"]),
            })
        except (KeyError, ValueError) as e:
            print(f"[warn] {path} bad row: {e}")
    return out

def hourly_candles(symbol, n=200):
    path = HOURLY_FILES.get(symbol)
    if not path or not os.path.exists(path):
        print(f"[warn] No hourly CSV for {symbol} ({path})")
        return []
    df = pd.read_csv(path).tail(n)
    out = []
    for _, row in df.iterrows():
        try:
            out.append({
                "time":  str(row["timestamp"]),
                "open":  float(row["open"]),
                "high":  float(row["high"]),
                "low":   float(row["low"]),
                "close": float(row["close"]),
            })
        except (KeyError, ValueError) as e:
            print(f"[warn] {path} bad row: {e}")
    return out

for symbol in predictions:
    predictions[symbol]["candles"] = daily_candles(symbol, 100)
    predictions[symbol]["candlesH1"] = hourly_candles(symbol, 200)

data_date = next(iter(predictions.values()))["date"]

doc_ref = db.collection("daily_predictions").document(data_date)
doc_ref.set({
    "predictions": predictions,
    "generated_at": firestore.SERVER_TIMESTAMP,
})

print(f"Wrote daily_predictions/{data_date}")
print(f"Daily candles: {[len(v.get('candles', [])) for v in predictions.values()]}")
print(f"Hourly candles: {[len(v.get('candlesH1', [])) for v in predictions.values()]}")