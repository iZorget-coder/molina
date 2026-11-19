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

# Map symbol → daily CSV for candle history
CANDLE_FILES = {
    "SPX":    "SP500_1d_data.csv",
    "NDX":    "NASDAQ_1d_data.csv",
    "DJI":    "DOW_1d_data.csv",
    "XAUUSD": "XAUUSD_1d_data.csv",
}

def last_candles(symbol, n=100):
    """Return the last `n` daily OHLC bars for a market as a list of dicts."""
    path = CANDLE_FILES.get(symbol)
    if not path or not os.path.exists(path):
        print(f"[warn] No CSV found for {symbol} ({path})")
        return []

    df = pd.read_csv(path).tail(n)
    candles = []
    for _, row in df.iterrows():
        try:
            candles.append({
                "date":  str(row["Date"]),
                "open":  float(row["Open"]),
                "high":  float(row["High"]),
                "low":   float(row["Low"]),
                "close": float(row["Close"]),
            })
        except (KeyError, ValueError) as e:
            print(f"[warn] Skipping malformed row in {path}: {e}")
    return candles

# Attach candle history to each market's prediction
for symbol in predictions:
    predictions[symbol]["candles"] = last_candles(symbol, 100)

# Use the data date from the predictions (not the run date)
data_date = next(iter(predictions.values()))["date"]

doc_ref = db.collection("daily_predictions").document(data_date)
doc_ref.set({
    "predictions": predictions,
    "generated_at": firestore.SERVER_TIMESTAMP,
})

print(f"Successfully wrote daily_predictions/{data_date}")
print(json.dumps(predictions, indent=2))