import numpy as np

# Function to preprocess the inputs and convert categorical variables into one-hot encoded format
def preprocess_input(age, hypertension, heart_disease, avg_glucose_level, bmi, gender, ever_married, work_type, residence_type, smoking_status):
    # Initialize a feature list of 16 features as per your model's input
    features = []
    # print(f"{age}, {hypertension}, {heart_disease}, {avg_glucose_level},"
        # f"{bmi}, {gender}, {ever_married}, {work_type}, {residence_type}, {smoking_status}")

    # Numeric variables
    features.append(age)
    features.append(hypertension)  # Hypertension: 0 or 1
    features.append(heart_disease)  # Heart disease: 0 or 1
    features.append(avg_glucose_level)  # Avg Glucose Level: float
    features.append(bmi)  # BMI: float

    # Gender: One-hot encoding
    if gender == "Male":
        features.extend([1, 0])  # Male: [1, 0]
    elif gender == "Other":
        features.extend([0, 1])  # Other: [0, 1]
    else:  # Female is implied to be [0, 0]
        features.extend([0, 0])

    # Ever Married: One-hot encoding (assuming "Yes" is 1, and "No" is 0)
    features.append(1 if ever_married == "Yes" else 0)

    # Work Type: One-hot encoding for 4 categories (Never_worked, Private, Self-employed, Children)
    work_types = ["Never worked", "Private", "Self employed", "Children"]
    work_type_encoded = [1 if work_type == wt else 0 for wt in work_types]
    features.extend(work_type_encoded)

    # Residence Type: One-hot encoding (Urban: 1, Rural: 0)
    features.append(1 if residence_type == "Urban" else 0)

    # Smoking Status: One-hot encoding for 3 categories (formerly smoked, never smoked, smokes)
    smoking_statuses = ["formerly smoked", "never smoked", "smokes"]
    smoking_status_encoded = [1 if smoking_status == s else 0 for s in smoking_statuses]
    features.extend(smoking_status_encoded)

    # Convert list to numpy array for model prediction
    return np.array(features)

if  __name__=="__main__":
    preprocess_input(
        age=65,
        hypertension=1,
        heart_disease=1,
        avg_glucose_level=180,
        bmi=32,
        gender="Male",
        ever_married="Yes",
        work_type="Private",
        residence_type=1,
        smoking_status="smokes"
    )
