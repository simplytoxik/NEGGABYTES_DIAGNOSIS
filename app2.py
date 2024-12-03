from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier  # Example model
import threading

app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data(filepath):
    # Load data
    data = pd.read_csv(filepath)
    # Example preprocessing
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    label_encoders = {}
    # Assuming 'gender' is a categorical column
    for column in ['gender']:
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_encoders[column] = le
    
    return X_scaled, y, scaler, label_encoders

# Train model
def train_model(X, y):
    model = RandomForestClassifier()  # Example model
    model.fit(X, y)
    return model

# Load data and train the model
filepath = 'riyal_data.csv'  # Ensure this file exists in the working directory
X, y, scaler, label_encoders = load_and_preprocess_data(filepath)
trained_model = train_model(X, y)

# Function to predict diagnosis
def predict_diagnosis(model, scaler, label_encoders, age, gender, height, weight, systolic, diastolic):
    # Prepare input data
    input_data = np.array([[age, gender, height, weight, systolic, diastolic]])
    # Scale input data
    input_scaled = scaler.transform(input_data)
    # Make prediction
    prediction = model.predict(input_scaled)
    # Decode prediction if necessary
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = int(data['age'])
    gender = data['gender']
    height = float(data['height'])
    weight = float(data['weight'])
    systolic = int(data['systolic'])
    diastolic = int(data['diastolic'])
    
    diagnosis = predict_diagnosis(trained_model, scaler, label_encoders, age, gender, height, weight, systolic, diastolic)
    
    return jsonify({'diagnosis': diagnosis})

if __name__ == '__main__':
    app.run(debug=True)