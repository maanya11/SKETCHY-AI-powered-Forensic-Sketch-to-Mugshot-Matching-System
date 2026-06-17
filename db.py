from pymongo import MongoClient
from .config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

suspects = db["suspects"]
search_logs = db["search_logs"]
