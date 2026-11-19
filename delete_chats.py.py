# delete_chats.py — runs in the GitHub Action after push_to_firestore.py
from firebase_admin import firestore
db = firestore.client()

users = db.collection('users').stream()
for user in users:
    chats = user.reference.collection('chats').stream()
    for chat in chats:
        chat.reference.delete()
print("Deleted all chats")