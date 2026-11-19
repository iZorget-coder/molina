import firebase_admin
from firebase_admin import credentials, firestore
import json, os

service_account_info = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred)
db = firestore.client()

users = db.collection('users').stream()
for user in users:
    chats_ref = user.reference.collection('chats')
    docs = chats_ref.stream()
    for doc in docs:
        doc.reference.delete()
print("Deleted all chat documents.")