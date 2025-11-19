# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
le_stage = data["le_stage"]
le_disease = data["le_disease"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    stage = request.form["stage"]

    # Encode stage
    stage_encoded = le_stage.transform([stage])[0]

    # Predict
    pred = model.predict([[stage_encoded]])[0]
    disease = le_disease.inverse_transform([pred])[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Disease: {disease}"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
