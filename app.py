from fastapi import FastAPI
import pickle
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.get("/predict")
def predict(hours: float):
    x = {"hours": hours}
    prediction = model.predict_one(x)
    return {"hours": hours, "prediction": prediction}