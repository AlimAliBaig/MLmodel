from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from HTML form
        cgpa = float(request.form["cgpa"])
        iq = float(request.form["iq"])
        profile_score = float(request.form["profile_score"])

        # Create input in same order as training:
        # cgpa, iq, profile_score
        input_data = np.array([
            [cgpa, iq, profile_score]
        ])

        # Prediction
        prediction = model.predict(input_data)[0]

        # Probability if available
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1] * 100
        else:
            probability = None

        if prediction == 1:
            result = "Student is likely to be PLACED"
        else:
            result = "Student is likely NOT to be PLACED"

        return render_template(
            "index.html",
            prediction=result,
            probability=probability,
            cgpa=cgpa,
            iq=iq,
            profile_score=profile_score
        )

    except Exception as e:
        return render_template(
            "index.html",
            error="Error: " + str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)