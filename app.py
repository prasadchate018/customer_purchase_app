import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sys
import types

# --- FIX FOR _LOSS UNPICKLING ERROR ---
# If the model was saved with an older/different sklearn version that references '_loss',
# we create a safe dummy module in memory so pickle.load() doesn't crash.
if '_loss' not in sys.modules:
    dummy_loss = types.ModuleType('_loss')
    sys.modules['_loss'] = dummy_loss

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
    </style>
""", unsafe_allow_init=True if "unsafe_allow_init" in globals() else False) # standard streamlit below:
""", unsafe_allow_html=True)

# Load the Model
@st.cache_resource
def load_model():
    with open('gradient_boosting.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

st.title("⚡ Gradient Boosting Classifier App")
st.markdown("Enter the required details below to generate real-time predictions.")
st.markdown("---")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

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
    input_data = pd.DataFrame(
        [[age, gender, review, education]], 
        columns=['age', 'gender', 'review', 'education']
    )
    
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
        
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
