import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('riyal_data.csv')

# Data Cleaning Steps
df.fillna(df.mean(), inplace=True)  # Fill missing values with the mean
df.drop_duplicates(inplace=True)
df['Gender'] = df['Gender'].str.strip().str.lower()
df['Age'] = df['Age'].astype(int)

# Encode categorical variables
label_encoders = {}
for col in ['Gender', 'Diagnosis', 'Address']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
features = ['Age', 'Gender', 'Height', 'Weight', 'Systolic', 'Diastolic']
target = 'Diagnosis'

X = df[features]
y = df[target]

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create a Random Forest model with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the model's performance
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Define a function to predict diagnosis for new patient data
def predict_diagnosis(age, gender, height_cm, weight, systolic, diastolic):
    gender_encoded = label_encoders['Gender'].transform([gender])[0]
    scaled_inputs = scaler.transform([[age, gender_encoded, height_cm, weight, systolic, diastolic]])
    prediction = best_model.predict(scaled_inputs)
    diagnosis = label_encoders['Diagnosis'].inverse_transform(prediction)[0]
    return diagnosis

# Function to get user input and predict diagnosis
def user_input():
    a = input("Enter your name: ")
    print(f"ENNE PODE {a} JI,")
    print("Please enter the following details:")
    age = int(input("Age: "))
    gender = input("Gender (Male/Female/Two in One/One in Two): ")
    height_cm = float(input("Height in cm: "))
    weight = float(input("Weight in kg: "))
    systolic = int(input("Systolic Blood Pressure: "))
    diastolic = int(input("Diastolic Blood Pressure: "))

    diagnosis = predict_diagnosis(age, gender, height_cm, weight, systolic, diastolic)
    print(f"Predicted Diagnosis: {diagnosis}")

# Run the input function
user_input()