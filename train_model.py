from river import linear_model
import pickle

model = linear_model.LinearRegression()

data = [
    ({"hours": 2}, 4),
    ({"hours": 3}, 6),
    ({"hours": 4}, 8),
    ({"hours": 5}, 10),
]

for x, y in data:
    model.learn_one(x, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")