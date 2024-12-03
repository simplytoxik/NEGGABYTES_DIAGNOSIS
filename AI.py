





#                                ███╗   ██╗███████╗ ██████╗  ██████╗  █████╗ ██████╗ ██╗   ██╗████████╗███████╗███████╗
#                                ████╗  ██║██╔════╝██╔════╝ ██╔════╝ ██╔══██╗██╔══██╗╚██╗ ██╔╝╚══██╔══╝██╔════╝██╔════╝
#                                ██╔██╗ ██║█████╗  ██║  ███╗██║  ███╗███████║██████╔╝ ╚████╔╝    ██║   █████╗  ███████╗
#                                ██║╚██╗██║██╔══╝  ██║   ██║██║   ██║██╔══██║██╔══██╗  ╚██╔╝     ██║   ██╔══╝  ╚════██║
#                                ██║ ╚████║███████╗╚██████╔╝╚██████╔╝██║  ██║██████╔╝   ██║      ██║   ███████╗███████║
#                                ╚═╝  ╚═══╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝    ╚═╝      ╚═╝   ╚══════╝╚══════╝
                                                                                      




import pandas as pd #Here for managing and working with data
import matplotlib.pyplot as plt #Here for inserting data in the graph
import seaborn as sns #Here for different graphs
import random #Here for generating a 1000 data with few lines of code
import re #Just here for Lambda Function
# from aidata import generated_p_data
# random.seed(42) TRIBUTE IS MUST , but change is very must

from aidata import generated_p_data
generated_p_data()

df = pd.DataFrame(generated_p_data())


# Handling Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True) #Filter for No age peoples
df['Email'].fillna('missing@email.com', inplace=True)  # Default for missing emails
df['Blood_Pressure'].fillna('120/80', inplace=True)  # Default value for blood pressure

# Clean Phone Numbers (Remove non-numeric characters) 
df['Phone'] = df['Phone'].apply(lambda x: re.sub(r'[^0-9]', '', x))

# Convert Heights to Centimeters
def height_to_cm(height_str):
    feet, inches = height_str.split("'")
    inches = inches.replace('"', '')
    return int(feet) * 30.48 + int(inches) * 2.54

df['Height_cm'] = df['Height'].apply(height_to_cm)


# Split Blood Pressure into Systolic and Diastolic
df[['Systolic', 'Diastolic']] = df['Blood_Pressure'].str.split('/', expand=True)
df['Systolic'] = df['Systolic'].astype(int)
df['Diastolic'] = df['Diastolic'].astype(int)

# # Normalize Diagnoses to English 
# df['Diagnosis'] = df['Diagnosis'].replace({
#     'ज्वर': 'Fever',  # Flaunting Dict Functions
#     'जॉन्डिस': 'Jaundice'
    
# })

# 6. Validate Age (removing patients with negative age)
df = df[df['Age'] > 0]

# Initialization command to set the graph mode
sns.set(style="darkgrid")


#-------------------------------------------------------------------------------------------------------------------------------------------------------


# Age wise Distribution of genders
# HISTPLOT
plt.figure(figsize=(12, 8))

# Create the histogram with KDE, and adjusting color and bins
ax = sns.histplot(df['Age'], kde=True, color='skyblue', bins=25, edgecolor='black', linewidth=1.2)

# Title of the graph
plt.title('Age Distribution of Patients', fontsize=18, fontweight='bold')

# X and Y axis labels with larger font size
plt.xlabel('Age', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')

# Increase font size of ticks for both x and y axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add gridlines for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# Adjust the layout to make sure everything fits
plt.tight_layout()

# Show the plot
plt.show()


#-------------------------------------------------------------------------------------------------------------------------------------------------------


# Diagnosis Count by Gender 
# COUNTPLOT
plt.figure(figsize=(14, 8))

# Creating the countplot with 'hue' for gender differentiation
ax = sns.countplot(x='Diagnosis', hue='Gender', data=df, palette='Set2', dodge=True)

# Title with larger font size and bold
plt.title('Diagnosis Count by Gender', fontsize=18, fontweight='bold')

# Label the axes with larger font size and bold
plt.xlabel('Diagnosis', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')

# Rotate the x-axis labels for better readability and adjust fontsize
plt.xticks(rotation=45, ha='right', fontsize=12)

# Customize the legend for better readability and positioning
plt.legend(title='Gender', fontsize=12, title_fontsize=14, loc='upper right')

# Add gridlines for better visibility of the counts
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# Remove top and right spines for a cleaner look
sns.despine()

# Adjust layout to ensure everything fits without overlap
plt.tight_layout()

# Show the plot
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------


# Blood Pressure Distribution (Systolic and Diastolic)
# BOXPLOT
plt.figure(figsize=(14, 8))

# Create the boxplot with a better color palette
ax = sns.boxplot(data=df[['Systolic', 'Diastolic']], palette='coolwarm', width=0.5)

# Add a title with larger font size and bold
plt.title('Blood Pressure Distribution (Systolic and Diastolic)', fontsize=18, fontweight='bold')

# Label the axes with larger font size and bold
plt.xlabel('Blood Pressure', fontsize=14, fontweight='bold')
plt.ylabel('Pressure Value (mmHg)', fontsize=14, fontweight='bold')

# Customizing tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add gridlines behind the plot to enhance readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# Remove top and right spines for a cleaner look
sns.despine()

# Adjust the layout to ensure it fits within the figure
plt.tight_layout()

plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------


# Height vs. Weight
# SCATTERPLOT
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Height_cm', y='Weight', data=df, hue='Gender', palette='muted', alpha=0.7, size='Age', sizes=(20, 200))
plt.title
plt.show()


#-------------------------------------------------------------------------------------------------------------------------------------------------------


# Diagnosis Count
# COUNTPLOT
sns.set(style="whitegrid", palette="muted")

plt.figure(figsize=(14, 8))
ax = sns.countplot(x='Diagnosis', data=df, palette='Set1')

plt.title('Number of Patients by Diagnosis', fontsize=18, fontweight='bold')
plt.xlabel('Diagnosis', fontsize=14, fontweight='bold')
plt.ylabel('Number of Patients', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
sns.despine()
plt.tight_layout()
plt.show()

print("THE PROJECT IS FINISHED\n")

print("BUT WE ARE NOT DONE YET!!!!!\n")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





                                                                          
#                                                █████  ██     ███    ███  ██████  ██████  ███████ ██      
#                                               ██   ██ ██     ████  ████ ██    ██ ██   ██ ██      ██      
#                                               ███████ ██     ██ ████ ██ ██    ██ ██   ██ █████   ██      
#                                               ██   ██ ██     ██  ██  ██ ██    ██ ██   ██ ██      ██      
#                                               ██   ██ ██     ██      ██  ██████  ██████  ███████ ███████ 
                                                           
                                                           





from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.ensemble import RandomForestClassifier  # Random Forest algorithm for classification
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding categorical data and scaling numerical data
from sklearn.metrics import accuracy_score, classification_report  # For evaluating the model(Used only while testing by me)
import numpy as np  # For numerical computations


# Encode features 
label_encoders = {}  # Dictionary to store LabelEncoder objects
for col in ['Gender', 'Diagnosis', 'Address']:
    le = LabelEncoder()  # Starting LabelEncoder
    df[col] = le.fit_transform(df[col])  # Fit and transform the column values to numeric
    label_encoders[col] = le  # Store the encoder

# Select features and target
features = ['Age', 'Gender', 'Height_cm', 'Weight', 'Systolic', 'Diastolic']
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
user_input()




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%