import firebase_admin
from firebase_admin import firestore
import os

if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

firebase_app = firebase_admin.initialize_app()
db = firestore.client()

class DatabaseClient:
    def __init__(self):
        self.games_ref = db.collection("games")
        self.friends_ref = db.collection("friends")
        self.users_games_ref = db.collection("users_games")
        self.interactions_ref = db.collection("interactions")
