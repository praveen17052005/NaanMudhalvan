from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load the saved model
model = joblib.load("model.pkl")

# Features your model expects
features = [
    "Age",
    "Health Concern",
    "Self Cooking",
    "Good Food quality",
    "Late Delivery",
    "More Offers and Discount"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Create DataFrame for prediction
        input_df = pd.DataFrame([data], columns=features)

        # Predict churn
        prediction = model.predict(input_df)[0]

        return jsonify({'prediction': str(prediction)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'prediction': 'Prediction failed.'})

if __name__ == '__main__':
    app.run(debug=True)
