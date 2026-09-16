import joblib
import numpy as np
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Model Prediction App", page_icon="🤖", layout="centered"
)

## 1. Load the Model Safely
@st.cache_resource
def load_model():
  try:
    # Replace 'model.pkl' with your actual model filename
    model = joblib.load("model.pkl")
    return model
  except Exception as e:
    st.error(
        "Error loading model file. This is usually caused by a scikit-learn"
        " version mismatch (e.g., model trained with v1.6.1 but running on an"
        " older version)."
    )
    st.exception(e)
    return None


model = load_model()

# App UI Header
st.title("Machine Learning Prediction App")
st.write(
    "Provide the input features below to get a prediction from your trained"
    " model."
)

if model is not None:
  # Example input fields matching features found in your model (age, gender, review, education)
  with st.form("prediction_form"):
    st.subheader("Input Features")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education = st.selectbox(
        "Education Level", ["High School", "Bachelor", "Master", "PhD"]
    )
    review = st.text_area("Review / Comments", "Type your review here...")

    submitted = st.form_submit_button("Predict")

    if submitted:
      try:
        # Format inputs according to how your pipeline/model expects them
        # Note: Adjust this preprocessing array to match your exact training features format
        input_data = np.array([[age]])  # Modify based on your feature requirements

        prediction = model.predict(input_data)
        st.success(f"Prediction Result: {prediction[0]}")
      except Exception as prediction_error:
        st.error(f"An error occurred during prediction: {prediction_error}")
else:
  st.warning(
      "Please fix the model loading issue (upgrade scikit-learn to match the"
      " training environment) to proceed."
  )
