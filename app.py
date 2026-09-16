import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Gradient Boosting Predictor",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.6rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model with Caching
@st.cache_resource
def load_model():
    with open('gradient_boosting.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

# Header Section
st.title("🎯 Gradient Boosting Classifier")
st.markdown("Provide your input details below to generate a real-time prediction from the deployed model.")
st.markdown("---")

# Input Form Layout
with st.form("prediction_form"):
    st.subheader("📝 Input Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", options=["No", "Yes", "Male", "Female"])
        
    with col2:
        education = st.selectbox("Education Level", options=["High School", "Bachelor", "Master", "PhD"])
        review = st.text_input("Review / Feedback", value="Great experience")
        
    submitted = st.form_submit_button("Run Prediction")

# Prediction Logic
if submitted:
    # Constructing dataframe matching the model's expected feature names: age, gender, review, education
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'review': [review],
        'education': [education]
    })
    
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        result_class = prediction[0]
        
        if str(result_class).lower() in ["yes", "1", "positive"]:
            st.success(f"**Prediction Outcome:** {result_class}")
        else:
            st.info(f"**Prediction Outcome:** {result_class}")
            
        if prediction_proba is not None:
            confidence = np.max(prediction_proba) * 100
            st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Note: Ensure that categorical inputs match the encoding format used during your model's training phase.")
