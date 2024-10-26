import streamlit as st
import pickle
import numpy as np
import streamlit.components.v1 as components
from helper_utils import preprocess_input

# Load the Random Forest model
def load_model():
    with open("E:\brain_stroke_prediction_2\brain_stroke_prediction_2\models\random_forest.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Function to predict if a person is likely to have a stroke
def predict_stroke(model, features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit App UI
def home_page():
    st.set_page_config(page_title="Brain Stroke Prediction", layout="wide")

    # Import custom CSS
    with open("static/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Header Section
    st.markdown("""
    <div class="header">
        <h1>Brain Stroke Prediction</h1>
        <h6>Predict stroke risk using machine learning by providing the necessary health parameters below.</h6>
    </div>
    """, unsafe_allow_html=True)

    # Embedding Custom JavaScript for Validation
    components.html("""
    <script>
        function validateGlucoseLevel() {
            var glucoseInput = document.getElementById("avgGlucoseLevel");
            var value = parseFloat(glucoseInput.value);
            if (isNaN(value) || value < 70 || value > 200) {
                glucoseInput.style.borderColor = "red";
                alert("Please enter a valid glucose level (between 70 and 200)");
            } else {
                glucoseInput.style.borderColor = "green";
            }
        }

        function validateBMI() {
            var bmiInput = document.getElementById("bmi");
            var value = parseFloat(bmiInput.value);
            if (isNaN(value) || value < 10 || value > 40) {
                bmiInput.style.borderColor = "red";
                alert("Please enter a valid BMI (between 10 and 40)");
            } else {
                bmiInput.style.borderColor = "green";
            }
        }
    </script>
    """, height=0)

    # Input Section
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        age = st.slider("Age", 0, 100, 25)
        avg_glucose_level = st.number_input("Average Glucose Level", value=100.0, format="%.2f", key="avgGlucoseLevel")
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

    with col2:
        hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1], key="hypertension")
        bmi = st.number_input("BMI", value=25.0, format="%.2f", key="bmi")
        work_type = st.selectbox("Work Type", ["Never worked", "Private", "Self employed", "Children"],
                                 key="workTypeField")

    with col3:
        heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
        gender = st.selectbox("Gender", ["Male", "Female"])
        residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Data Processing
    input_features = preprocess_input(
        age=age,
        hypertension=1 if hypertension == "Yes" else 0,
        heart_disease=1 if heart_disease == "Yes" else 0,
        avg_glucose_level=avg_glucose_level,
        bmi=bmi,
        gender=gender,
        ever_married=ever_married,
        work_type=work_type,
        residence_type=residence_type,
        smoking_status=smoking_status
    )

    model = load_model()

    button = st.button("Predict", type="primary")
    # Prediction Button
    if button:
        prediction = predict_stroke(model, input_features)
        if prediction == 1:
            st.error("Warning: The person is likely to have a stroke.", icon="⚠️")
        else:
            st.success("The person is not likely to have a stroke.", icon="✅")

    # Footer Section
    st.markdown("""
    <footer>
        <p>Powered by <strong>Random Forest</strong> | Keerthy VB</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    home_page()
