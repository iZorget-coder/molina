# push_to_firestore.py
import firebase_admin
from firebase_admin import credentials, firestore
import json
import os
from datetime import datetime

# Load the service account from the environment variable
service_account_info = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred)

db = firestore.client()
today = datetime.utcnow().strftime('%Y-%m-%d')

# --- !!! THIS IS A PLACEHOLDER !!! ---
# You must modify your 'market data.ipynb' to save its final
# predictions (direction, confidence, etc.) to a JSON file.
# This script then reads that file.
# Example: predictions = {"SPX": {"bias": "UP", "confidence": 0.62}, ...}

with open('predictions.json', 'r') as f:
    predictions = json.load(f)

# Write to a shared document that all users can read
doc_ref = db.collection('daily_predictions').document(today)
doc_ref.set({
    'predictions': predictions,
    'generated_at': firestore.SERVER_TIMESTAMP
})

print(f"Successfully wrote predictions for {today}")