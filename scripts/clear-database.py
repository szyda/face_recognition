from pymongo import MongoClient
from bson.binary import Binary
import json
from dotenv import load_dotenv
import os


load_dotenv(dotenv_path='../config.env')
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")

client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

collection.delete_many({})