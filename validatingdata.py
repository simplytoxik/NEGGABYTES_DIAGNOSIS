
    # # Handling Missing Values
    # df['Age'].fillna(df['Age'].median(), inplace=True) #Filter for No age peoples
    # df['Email'].fillna('missing@email.com', inplace=True)  # Default for missing emails
    # df['Blood_Pressure'].fillna('120/80', inplace=True)  # Default value for blood pressure

    # # Clean Phone Numbers (Remove non-numeric characters) 
    # df['Phone'] = df['Phone'].apply(lambda x: re.sub(r'[^0-9]', '', x))

    # # Convert Heights to Centimeters
    # def height_to_cm(height_str):
    #     feet, inches = height_str.split("'")
    #     inches = inches.replace('"', '')
    #     return int(feet) * 30.48 + int(inches) * 2.54

    # df['Height_cm'] = df['Height'].apply(height_to_cm)


    # # Split Blood Pressure into Systolic and Diastolic
    # df[['Systolic', 'Diastolic']] = df['Blood_Pressure'].str.split('/', expand=True)
    # df['Systolic'] = df['Systolic'].astype(int)
    # df['Diastolic'] = df['Diastolic'].astype(int)

    # # # Normalize Diagnoses to English 
    # # df['Diagnosis'] = df['Diagnosis'].replace({
    # #     'ज्वर': 'Fever',  # Flaunting Dict Functions
    # #     'जॉन्डिस': 'Jaundice'
        
    # # })

    # # 6. Validate Age (removing patients with negative age)
    # df = df[df['Age'] > 0]
