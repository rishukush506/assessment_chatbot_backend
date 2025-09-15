from pymongo import MongoClient, ReturnDocument
import dotenv
import os
import pandas as pd

dotenv.load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI,tls=True, tlsAllowInvalidCertificates=True)



db = client["chatbot_db"]
messages = db["messages_user_testing"]
personas = db["persona_user_testing"]
# messages_2 = db["messages"]
# personas_2 = db["persona"]

# Fetch the list of all user_ids from personas collection
def get_all_user_ids():
    user_ids = personas.distinct("user_id")
    return user_ids

def get_user_chats(user_id: str):
    """
    Fetch chat history for a given user_id
    """
    chats = list(messages.find({"user_id": user_id}))  # Exclude _id
    return chats

def get_user_persona(user_id: str):
    """
    Fetch persona for a given user_id
    """
    persona = personas.find({"user_id": user_id})
    return persona



user_id_list = get_all_user_ids()

# #  Fetch chat history for user_id_list users and append to results
chats = []

chats.append(get_user_chats("user73"))

# for user in user_id_list:
#     result=get_user_chats(user)
#     results.append(result)

df=pd.DataFrame([item for sublist in chats for item in sublist])
print(df.shape)
df.to_excel(f"temp_chat_history_all_users.xlsx", index=False)

results_persona = []
results_persona.append(get_user_persona("user73"))

# for user in user_id_list:
#     result=get_user_persona(user)
#     results_persona.append(result)

df_persona=pd.DataFrame([item for sublist in results_persona for item in sublist])
print(df_persona.shape)
df_persona.to_excel(f"temp_persona_all_users.xlsx", index=False)




# print(result)