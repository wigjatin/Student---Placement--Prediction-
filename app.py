import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('model.pkl')

# Page configuration
st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="wide")

# Custom CSS styles
st.markdown("""
    <style>
        body {
            background-color: #f0f8f1;  /* Soft green background */
        }
        .main-container {
            background-color: white;
            border-radius: 20px;
            padding: 2.5rem;
            margin: 2rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #388e3c;  /* Green color */
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #2c6e29;  /* Darker green on hover */
        }
        h1 {
            color: #388e3c;  /* Green header */
            text-align: center;
            font-size: 3em;
            margin-bottom: 1rem;
            font-weight: bold;
        }
        h3 {
            color: #388e3c;
            margin-top: 2rem;
        }
        label {
            font-weight: 600 !important;
        }
        .stRadio>label {
            color: #388e3c;  /* Green labels */
        }
        .stNumberInput>label {
            color: #388e3c;  /* Green labels */
        }
        /* Adjust the heading to avoid any box or background around it */
        .stTitle {
            margin-top: 0;
            padding: 0;
            background: none;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🎓 Placement Prediction System")
st.markdown("<h3>📊 Predict candidate placement outcome using academic performance and profile information.</h3>", unsafe_allow_html=True)

with st.form("placement_form"):
    st.subheader("📋 Enter Candidate Details")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio("Gender", ["Male", "Female"])
        ssc_b = st.radio("Senior School Board", ["CBSE", "Others"])
        hsc_s = st.selectbox("High School Stream", ["Science and Technology", "Commerce and Management", "Arts"])
        ssc_p = st.number_input("Senior School Percentage", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        hsc_p = st.number_input("High School Percentage", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
    
    with col2:
        hsc_b = st.radio("High School Board", ["CBSE", "Others"])
        degree_t = st.selectbox("Degree Type", ["Science and Technology", "Commerce and Management", "Others"])
        degree_p = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        workex = st.radio("Work Experience", ["Yes", "No"])
        etest_p = st.number_input("E-Test Percentage", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        specialisation = st.radio("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])
        mba_p = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

    submit_button = st.form_submit_button(label="🔍 Predict Placement")

if submit_button:
    input_data = pd.DataFrame({
        'gender': 1 if gender == 'Male' else 0,
        'ssc_p': ssc_p,
        'ssc_b': 1 if ssc_b == 'CBSE' else 0,
        'hsc_p': hsc_p,
        'hsc_b': 1 if hsc_b == 'CBSE' else 0,
        'hsc_s': {'Science and Technology': 2, 'Commerce and Management': 1, 'Arts': 0}[hsc_s],
        'degree_p': degree_p,
        'degree_t': {'Science and Technology': 2, 'Commerce and Management': 1, 'Others': 0}[degree_t],
        'workex': 1 if workex == 'Yes' else 0,
        'etest_p': etest_p,
        'specialisation': 1 if specialisation == 'Mkt&HR' else 0,
        'mba_p': mba_p
    }, index=[0])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("📈 Prediction Result")
    if pred == 1:
        st.success(f"✅ The candidate is likely to be placed!\n**Placement Probability:** {prob:.2%}")
    else:
        st.error("❌ The candidate is not likely to be placed.")
        st.info(f"**Placement Probability:** {prob:.2%}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Crafted with ❤️ using Streamlit & Scikit-learn")
