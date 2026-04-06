from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')  # ✅ REQUIRED for deployment
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load dataset
data = pd.read_csv("data.csv")

def extract_features(data):
    return data["brain_volume"] * (1 + data["heart_risk"])

def compute_velocity(values):
    return np.diff(values)

@app.route("/", methods=["GET", "POST"])
def home():
    global data

    # ✅ Handle user input safely
    if request.method == "POST":
        try:
            new_time = len(data) + 1
            brain = float(request.form["brain"])
            heart = float(request.form["heart"])

            new_row = pd.DataFrame({
                "time": [new_time],
                "brain_volume": [brain],
                "heart_risk": [heart],
                "age": [65]
            })

            data = pd.concat([data, new_row], ignore_index=True)

        except:
            return "Invalid input! Please enter valid numbers."

    features = extract_features(data)
    velocity = compute_velocity(features)

    X, y = [], []

    for i in range(1, len(features)):
        X.append([features.iloc[i], data["heart_risk"].iloc[i], velocity[i-1]])

        # ✅ Improved labeling (ensures 2 classes)
        if data["brain_volume"].iloc[i] < 90 or data["heart_risk"].iloc[i] > 0.6:
            y.append(1)
        else:
            y.append(0)

    # ✅ Prevent ML crash
    if len(set(y)) < 2:
        return "Not enough variation in data. Add more diverse inputs."

    model = LogisticRegression()
    model.fit(X, y)

    test_input = [[features.iloc[-1], data["heart_risk"].iloc[-1], velocity[-1]]]

    # 🔥 Probability prediction
    prob = model.predict_proba(test_input)[0][1]
    avg_velocity = np.mean(velocity)

    # 🔥 Risk levels
    if prob > 0.7:
        result = "High Risk Alzheimer's"
    elif prob > 0.4:
        result = "Moderate Risk"
    else:
        result = "Low Risk"

    explanation = f"""
    Risk Probability: {round(prob*100,2)}%
    Progression Speed: {round(avg_velocity,3)}
    """

    # 📊 Create graphs safely
    os.makedirs("static", exist_ok=True)

    # Brain graph
    plt.figure()
    plt.plot(data["time"], data["brain_volume"], marker='o')
    plt.xlabel("Time")
    plt.ylabel("Brain Volume")
    plt.title("Brain Volume Over Time")
    plt.savefig("static/brain.png")
    plt.close()

    # Heart graph
    plt.figure()
    plt.plot(data["time"], data["heart_risk"], marker='o')
    plt.xlabel("Time")
    plt.ylabel("Heart Risk")
    plt.title("Heart Risk Over Time")
    plt.savefig("static/heart.png")
    plt.close()

    # 🔮 Future prediction
    future_time = data["time"].iloc[-1] + 1
    future_brain = data["brain_volume"].iloc[-1] + avg_velocity
    future_text = f"Predicted Brain Volume at time {future_time}: {round(future_brain,2)}"

    return render_template(
        "index.html",
        result=result,
        explanation=explanation,
        future=future_text
    )

# ✅ IMPORTANT for deployment
if __name__ == "__main__":
    app.run(debug=True)