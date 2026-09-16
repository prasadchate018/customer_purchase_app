import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Gradient Boosting Predictor",
    page_icon="⚡",
    layout="centered"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
        color: white;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load the Model
@st.cache_resource
def load_model():
    with open('gradient_boosting.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

# App Header
st.title("⚡ Gradient Boosting Classifier App")
st.markdown("Enter the required details below to generate real-time predictions from your model.")
st.markdown("---")

# Input Form Layout
with st.form("prediction_form"):
    st.subheader("📋 Input Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        
    with col2:
        education = st.selectbox("Education Level", options=["High School", "Bachelor", "Master", "PhD", "Other"])
        review = st.selectbox("Review Sentiment", options=["Positive", "Neutral", "Negative"])
        
    st.markdown("")
    submitted = st.form_submit_button("Run Prediction")

# Prediction Execution
if submitted:
    # Assemble input DataFrame matching feature names in the model
    input_data = pd.DataFrame(
        [[age, gender, review, education]], 
        columns=['age', 'gender', 'review', 'education']
    )
    
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
        
        # Map prediction result to class labels if available
        classes = getattr(model, "classes_", ["No", "Yes"])
        predicted_class = classes[prediction[0]] if len(classes) > prediction[0] else prediction[0]
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        if str(predicted_class).lower() in ["yes", "1", "true", "positive"]:
            st.success(f"**Prediction:** {predicted_class} 🎉")
        else:
            st.info(f"**Prediction:** {predicted_class} ℹ️")
            
        if prediction_proba is not None:
            confidence = np.max(prediction_proba) * 100
            st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.warning("Note: If your model was trained on encoded numerical categories rather than raw strings, you may need to map the categorical inputs to match your training encoders.")
