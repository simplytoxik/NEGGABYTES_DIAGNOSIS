import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def load_and_preprocess_data(filepath):
    """
    Load data from a CSV file and preprocess it for modeling.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        X (DataFrame): Preprocessed feature data.
        y (Series): Encoded target labels.
        scaler (StandardScaler): Scaler for numerical features.
        label_encoders (dict): LabelEncoders for categorical features.
    """
    try:
        # Load dataset
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Please ensure it's in the same directory as this script.")
        exit(1)

    # Encode categorical features
    label_encoders = {}
    for col in ['Gender', 'Diagnosis', 'Address']:
        if col in df:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            print(f"Warning: Column '{col}' not found in the dataset.")

    # Select features and target
    features = ['Age', 'Gender', 'Height', 'Weight', 'Systolic', 'Diastolic']
    target = 'Diagnosis'

    # Check for missing columns
    if not all(col in df for col in features + [target]):
        print("Error: Missing required columns in the dataset. Please check the file format.")
        exit(1)

    X = df[features]
    y = df[target]

    # Scale numerical features
    scaler = StandardScaler()
    X[features] = scaler.fit_transform(X[features])

    return X, y, scaler, label_encoders


def train_model(X, y):
    """
    Train a Random Forest model.

    Parameters:
        X (DataFrame): Feature data.
        y (Series): Target labels.

    Returns:
        model (RandomForestClassifier): Trained classifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return model


def predict_diagnosis(model, scaler, label_encoders, age, gender, height, weight, systolic, diastolic):
    """
    Predict diagnosis for a new patient.

    Parameters:
        model (RandomForestClassifier): Trained classifier.
        scaler (StandardScaler): Scaler for feature standardization.
        label_encoders (dict): Encoders for categorical features.
        age (int): Patient's age.
        gender (str): Patient's gender.
        height (float): Patient's height in cm.
        weight (float): Patient's weight in kg.
        systolic (int): Systolic blood pressure.
        diastolic (int): Diastolic blood pressure.

    Returns:
        str: Predicted diagnosis.
    """
    try:
        # Encode categorical input
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
    except KeyError:
        print("Invalid gender entered. Please try again.")
        return None

    # Scale input features
    inputs = [[age, gender_encoded, height, weight, systolic, diastolic]]
    scaled_inputs = scaler.transform(inputs)

    # Predict diagnosis
    prediction = model.predict(scaled_inputs)
    return label_encoders['Diagnosis'].inverse_transform(prediction)[0]


def get_user_input():
    """
    Collect user inputs with validation and return them.
    """
    print("\nPlease enter the following details:")
    try:
        name = input("Enter your name: ")
        print(f"\nWelcome, {name}!")

        # Validate age
        age = int(input("Age (1-120): "))
        if not (1 <= age <= 120):
            raise ValueError("Age must be between 1 and 120.")

        # Validate gender
        gender = input("Gender (Male/Female): ").strip().capitalize()
        if gender not in ['Male', 'Female']:
            raise ValueError("Invalid gender. Enter 'Male' or 'Female'.")

        # Validate height
        height = float(input("Height (in cm, e.g., 170): "))
        if height <= 0:
            raise ValueError("Height must be a positive number.")

        # Validate weight
        weight = float(input("Weight (in kg, e.g., 65): "))
        if weight <= 0:
            raise ValueError("Weight must be a positive number.")

        # Validate blood pressure
        systolic = int(input("Systolic Blood Pressure (e.g., 120): "))
        diastolic = int(input("Diastolic Blood Pressure (e.g., 80): "))

        return age, gender, height, weight, systolic, diastolic
    except ValueError as e:
        print(f"Input Error: {e}")
        return get_user_input()


def main():
    # Filepath for the dataset
    filepath = 'riyal_data.csv'

    # Load and preprocess data
    X, y, scaler, label_encoders = load_and_preprocess_data(filepath)

    # Train the model
    trained_model = train_model(X, y)

    # Predict for user inputs
    user_inputs = get_user_input()
    diagnosis = predict_diagnosis(trained_model, scaler, label_encoders, *user_inputs)
    if diagnosis:
        print(f"\nPredicted Diagnosis: {diagnosis}")
    else:
        print("Prediction failed due to invalid inputs.")


if __name__ == "__main__":
    main()
