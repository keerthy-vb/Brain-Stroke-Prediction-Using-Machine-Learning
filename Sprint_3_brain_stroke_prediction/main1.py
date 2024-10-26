import streamlit as st
import pickle
import numpy as np
import streamlit.components.v1 as components

# Function to load the model
def load_model():
    with open("E:/brain_stroke_prediction_2/models/random_forest.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Function to preprocess inputs
def preprocess_input(age, hypertension, heart_disease, avg_glucose_level, bmi, gender, ever_married, work_type, residence_type, smoking_status):
    features = []
    # Numeric variables
    features.append(age)
    features.append(hypertension)
    features.append(heart_disease)
    features.append(avg_glucose_level)
    features.append(bmi)

    # Gender one-hot encoding
    if gender == "Male":
        features.extend([1, 0]) 
    elif gender == "Other":
        features.extend([0, 1]) 
    else:  # Female
        features.extend([0, 0])

    # Ever Married one-hot encoding
    features.append(1 if ever_married == "Yes" else 0)

    # Work Type one-hot encoding
    work_types = ["Never worked", "Private", "Self employed", "Children"]
    features.extend([1 if work_type == wt else 0 for wt in work_types])

    # Residence Type one-hot encoding
    features.append(1 if residence_type == "Urban" else 0)

    # Smoking Status one-hot encoding
    smoking_statuses = ["formerly smoked", "never smoked", "smokes"]
    features.extend([1 if smoking_status == s else 0 for s in smoking_statuses])

    return np.array(features)

# Prediction function
def predict_stroke(model, features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit App
def main():
    st.set_page_config(page_title="Brain Stroke Prediction", layout="wide")

    # Custom CSS styling
    st.markdown("""
    <style>
        /* Hide Streamlit components */
        #MainMenu, footer, .stAppHeader {visibility: hidden;}

        /* Background styling */
        .stApp {
            background-image: url('brain_stroke_prediction_2/download.jpg');
            background-size: cover;
            background-attachment: fixed;
        }

        /* Container styling */
        .form-container {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 40px;
            border-radius: 15px;
            max-width: 900px;
            margin: auto;
        }

        /* Header styling */
        h1 {
            font-size: 3em;
            color: #2356ff;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0;
        }

        h6 {
            color: #10182c;
            text-align: center;
            margin-bottom: 40px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("""
    <div class="form-container">
        <h1>Brain Stroke Prediction</h1>
        <h6>Predict stroke risk using machine learning by providing the necessary health parameters below.</h6>
    """, unsafe_allow_html=True)

    # Form fields for input
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 0, 100, 25)
        avg_glucose_level = st.number_input("Average Glucose Level", value=100.0, format="%.2f")
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

    with col2:
        hypertension = st.selectbox("Hypertension", [0, 1])
        bmi = st.number_input("BMI", value=25.0, format="%.2f")
        work_type = st.selectbox("Work Type", ["Never worked", "Private", "Self employed", "Children"])

    with col3:
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])

    # Preprocess input and load model
    input_features = preprocess_input(
        age=age,
        hypertension=hypertension,
        heart_disease=heart_disease,
        avg_glucose_level=avg_glucose_level,
        bmi=bmi,
        gender=gender,
        ever_married=ever_married,
        work_type=work_type,
        residence_type=residence_type,
        smoking_status=smoking_status
    )
    model = load_model()

    # Prediction button
    button = st.button("Predict Stroke Risk")
    if button:
        prediction = predict_stroke(model, input_features)
        if prediction == 1:
            st.error("⚠️ Warning: The person is likely to have a stroke.")
        else:
            st.success("✅ The person is not likely to have a stroke.")

    # Footer Section
    st.markdown("""
        <footer style="text-align:center; margin-top: 30px;">
            <p>Powered by <strong>Random Forest</strong> | Keerthy VB</p>
        </footer>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
