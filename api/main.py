from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/heart_treeModel.pkl')

# Define expected feature order (13 features)
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak',
            'slope', 'ca', 'thal']

@app.route('/')
def home():
    return "Heart Disease Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        input_data = [data[feature] for feature in FEATURES]
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        return jsonify({
            "prediction": int(prediction),
            "result": "Heart Disease" if prediction == 1 else "No Heart Disease"
        })

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
