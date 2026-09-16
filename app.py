import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration & Styling
st.set_page_config(
    page_title="Gradient Boosting Predictor", page_icon="⚡", layout="centered"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #2e7d32;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Load the Model
@st.cache_resource
def load_model():
    with open("gradient_boosting.pkl", "rb") as f:
        model = pickle.load(f)
    return model


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# App Header
st.title("⚡ Gradient Boosting Classifier")
st.markdown("Provide the information below to get your prediction.")
st.write("---")

# Input Form Layout
with st.form("prediction_form"):
    st.subheader("Input Features")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=1, max_value=120, value=30, step=1
        )
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])

    with col2:
        education = st.selectbox(
            "Education Level",
            options=["High School", "Bachelor", "Master", "PhD"],
        )
        review = st.selectbox("Review Status", options=["No", "Yes"])

    submitted = st.form_submit_button("Predict Now")

# Prediction Logic
if submitted:
    # Construct DataFrame matching the exact feature names: ['age', 'gender', 'review', 'education']
    input_data = pd.DataFrame(
        [[age, gender, review, education]],
        columns=["age", "gender", "review", "education"],
    )

    try:
        prediction = model.predict(input_data)
        prediction_proba = (
            model.predict_proba(input_data)
            if hasattr(model, "predict_proba")
            else None
        )

        st.write("---")
        st.subheader("Result")

        result_text = (
            str(prediction[0])
            if isinstance(prediction, (list, np.ndarray))
            else str(prediction)
        )

        st.markdown(
            f'<div class="prediction-box">Prediction: {result_text}</div>',
            unsafe_allow_html=True,
        )

        if prediction_proba is not None:
            confidence = np.max(prediction_proba) * 100
            st.info(f"Confidence Score: **{confidence:.2f}%**")

    except Exception as e:
        st.error(
            f"An error occurred during prediction. Ensure your model pipeline handles string categorical encoding properly. Details: {e}"
        )
