from model import Model
from pymongo import MongoClient


model = Model()
predictions = model.train()
predictions = [{'date': int(key), 'price': float(value)} for key, value in predictions.items()]

client = MongoClient()
db = client.fpa
db.predictions.delete_many({})
db.predictions.insert_many(predictions)
