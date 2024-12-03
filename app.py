from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from final_model import train_model, predict_diagnosis, load_and_preprocess_data

app = Flask(__name__)

# Load and preprocess data when the app starts
filepath = 'riyal_data.csv'
X, y, scaler, label_encoders = load_and_preprocess_data(filepath)
trained_model = train_model(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extracting data from the request
    age = data.get('age')
    gender = data.get('gender')
    height = data.get('height')
    weight = data.get('weight')
    systolic = data.get('systolic')
    diastolic = data.get('diastolic')

    # Validate inputs
    if None in (age, gender, height, weight, systolic, diastolic):
        return jsonify({'error': 'Missing input data'}), 400

    # Predict diagnosis using the loaded model
    diagnosis = predict_diagnosis(trained_model, scaler, label_encoders, age, gender, height, weight, systolic, diastolic)
    
    return jsonify({'diagnosis': diagnosis})

if __name__ == '__main__':
    app.run(debug=True)