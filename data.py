from pymongo import MongoClient, ReturnDocument
import dotenv
import os

dotenv.load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI,tls=True, tlsAllowInvalidCertificates=True)

db = client["chatbot_db"]
messages = db["messages"]

def get_user_chats(user_id: str):
    """
    Fetch chat history for a given user_id
    """
    chats = list(messages.find({"user_id": user_id}))  # Exclude _id
    return chats

result=get_user_chats("user88")

print(result)