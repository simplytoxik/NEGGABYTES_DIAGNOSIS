import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

def load_and_clean_data(filepath):
    """Load and clean the dataset."""
    df = pd.read_csv(filepath)
    df.fillna(df.mean(), inplace=True)  # Fill missing values with the mean
    df.drop_duplicates(inplace=True)
    df['Gender'] = df['Gender'].str.strip().str.lower()
    df['Age'] = df['Age'].astype(int)
    return df

def encode_categorical_variables(df):
    """Encode categorical variables and return label encoders."""
    label_encoders = {}
    for col in ['Gender', 'Diagnosis', 'Address']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def preprocess_data(df):
    """Preprocess the data and handle class imbalance."""
    features = ['Age', 'Gender', 'Height', 'Weight', 'Systolic', 'Diastolic']
    target = 'Diagnosis'
    
    X = df[features]
    y = df[target]

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the Random Forest model with hyperparameter tuning."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return y_pred

def predict_diagnosis(model, label_encoders, age, gender, height_cm, weight, systolic, diastolic):
    """Predict diagnosis for new patient data."""
    gender_encoded = label_encoders['Gender'].transform([gender])[0]
    scaled_inputs = np.array([[age, gender_encoded, height_cm, weight, systolic, diastolic]])
    prediction = model.predict(scaled_inputs)
    diagnosis = label_encoders['Diagnosis'].inverse_transform(prediction)[0]
    return diagnosis

def user_input(label_encoders, model):
    """Get user input and predict diagnosis."""
    name = input("Enter your name: ")
    print(f"Hello {name},")
    print("Please enter the following details:")
    
    try:
        age = int(input("Age: "))
        gender = input("Gender (Male/Female/Two in One/One in Two): ")
        height_cm = float(input("Height in cm: "))
        weight = float(input("Weight in kg: "))
        systolic = int(input("Systolic Blood Pressure: "))
        diastolic = int(input("Diastolic Blood Pressure: "))

        diagnosis = predict_diagnosis(model, label_encoders, age, gender, height_cm, weight, systolic, diastolic)
        print(f"Predicted Diagnosis: {diagnosis}")

    except ValueError:
        print("Invalid input. Please enter the correct data types.")

def main():
    """Main function to run the workflow."""
    df = load_and_clean_data('riyal_data.csv')
    df, label_encoders = encode_categorical_variables(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    best_model = train_model(X_train, y_train)
    evaluate_model(best_model, X_test, y_test)
    user_input(label_encoders, best_model)

# Run the main function
if __name__ == "__main__":
    main()