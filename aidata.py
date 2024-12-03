import pandas as pd  
import random  
from faker import Faker  
from datetime import datetime, timedelta  

# Initialize Faker and set a seed for reproducibility  
fake = Faker('en_IN')  # Using Indian locale for names and addresses  

# Parameters  
num_records = 10000  

# Gender options  
genders = ['Non-Binary', 'Female', 'Male', 'Gender Fluid','Agender','Bigender']  

# Sample diagnoses  
diagnoses = [  
    'Mithi Bimaari', 'Bin Mausam Sardi', 'Heavy Brain', 'Dancing stomach',  
    'Dispressure', 'Elephant Foot', 'Iron-Less Body', 'Crying Back','Money ILL','Jaundice'   
]  

# Function to generate random patient data  
def generate_patient_data(num):  
    data = []  
    for patient_id in range(num):  
        name = fake.name()  
        age = random.randint(18, 80)  
        gender = random.choice(genders)  
        address = fake.address().replace('\n', ', ')  
        phone = fake.phone_number()  
        email = fake.email()  
        date_of_visit = fake.date_between(start_date='-1y', end_date='today').strftime('%Y-%m-%d')  
        diagnosis = random.choice(diagnoses)  
        blood_pressure = f"{random.randint(110, 160)}/{random.randint(70, 100)}"  
        height = random.randint(150, 200)  # Height in cm  
        weight = random.randint(45, 100)    # Weight in kg  

        data.append({  
            "Patient_ID": f"P{patient_id + 1:04d}",  
            "Name": name,  
            "Age": age,  
            "Gender": gender,  
            "Address": address,  
            "Phone": phone,  
            "Email": email,  
            "Date_of_visit": date_of_visit,  
            "Diagnosis": diagnosis,  
            "Blood_Pressure": blood_pressure,  
            "Height": height,  
            "Weight": weight  
        })  

    return pd.DataFrame(data)  

# Generate the DataFrame  
patient_data_df = generate_patient_data(num_records)  

# Save to CSV  
patient_data_df.to_csv('riyal_data.csv', index=False)  

print("Synthetic patient data generated and saved as 'riyal_data.csv'")
