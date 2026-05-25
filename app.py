from fastapi import FastAPI
import pickle

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.get("/predict")
def predict(hours: float):
    x = {"hours": hours}
    prediction = model.predict_one(x)
    return {"hours": hours, "prediction": prediction}