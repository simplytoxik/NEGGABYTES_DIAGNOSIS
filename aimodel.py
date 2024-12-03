



                                                                          
#                                                █████  ██     ███    ███  ██████  ██████  ███████ ██      
#                                               ██   ██ ██     ████  ████ ██    ██ ██   ██ ██      ██      
#                                               ███████ ██     ██ ████ ██ ██    ██ ██   ██ █████   ██      
#                                               ██   ██ ██     ██  ██  ██ ██    ██ ██   ██ ██      ██      
#                                               ██   ██ ██     ██      ██  ██████  ██████  ███████ ███████ 
                                                           
                                                           


def model():

    import pandas as pd #Here for managing and working with data
    from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
    from sklearn.ensemble import RandomForestClassifier  # Random Forest algorithm for classification
    from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding categorical data and scaling numerical data
    from sklearn.metrics import accuracy_score, classification_report  # For evaluating the model(Used only while testing by me)
    import numpy as np  # For numerical computations
    import re #Just here for Lambda Function


    df = pd.read_csv('riyal_data.csv')
    
    # Encode features 
    label_encoders = {}  # Dictionary to store LabelEncoder objects
    for col in ['Gender', 'Diagnosis', 'Address']:
        le = LabelEncoder()  # Starting LabelEncoder
        df[col] = le.fit_transform(df[col])  # Fit and transform the column values to numeric
        label_encoders[col] = le  # Store the encoder

    # Select features and target
    features = ['Age', 'Gender', 'Height', 'Weight', 'Systolic', 'Diastolic']
    target = 'Diagnosis'  # The column we want to predict

    X = df[features]  # Extract feature data from the DataFrame
    y = df[target]  # Extract target data from the DataFrame

    # Scale numerical features to standardize their range
    scaler = StandardScaler()  # Initialize the scaler       #TO GET INPUTS IN LIMIT#
    X = scaler.fit_transform(X)  # Fit the scaler and transform the feature data

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  
    # 80% data for training, 20% for testing; random_state ensures reproducibility(Random state removed to see variety)

    # Train the model using a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100)  # Initialize the classifier with 100 trees
    model.fit(X_train, y_train)  # Train the model on the training data

    # # TO Evaluate the model's performance
    y_pred = model.predict(X_test)  # Make predictions on the test data
    print("Accuracy:", accuracy_score(y_test, y_pred))  # Print the accuracy of the model
    print("Classification Report:\n", classification_report(y_test, y_pred))  # Print detailed evaluation metrics




    #                                                                  Model Training Complete




    # Define a function to predict diagnosis for new patient data
    def predict_diagnosis(age, gender, height_cm, weight, systolic, diastolic):#                                                         
        # Encode the gender input to match training data
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        # Combine inputs into a single array and scale them(Making the inputs in limit)
        scaled_inputs = scaler.transform([[age, gender_encoded, height_cm, weight, systolic, diastolic]])
        # Predict the diagnosis using the trained model
        prediction = model.predict(scaled_inputs)
        # Removing the encoding
        diagnosis = label_encoders['Diagnosis'].inverse_transform(prediction)[0]
        return diagnosis

    # Function to get user input and predict diagnosis
    def user_input():
        a=input("Enter your name:")
        print(f"ENNE PODE {a} JI,")
        print("Please enter the following details:")  
        age = int(input("Age: "))  
        gender = input("Gender (Male/Female/Two in One/One in Two): ")  
        height_cm = float(input("Height in cm: "))  
        weight = float(input("Weight in kg: "))  
        systolic = int(input("Systolic Blood Pressure: ")) 
        diastolic = int(input("Diastolic Blood Pressure: "))  

        # Predict the diagnosis using the entered details
        diagnosis = predict_diagnosis(age, gender, height_cm, weight, systolic, diastolic)
        print(f"Predicted Diagnosis: {diagnosis}") 

    # Run the input function
    
    return user_input
model()