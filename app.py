import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')

st.set_page_config(page_title="Placement Predictor", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background-color: #f8fafc;
        }
        
        .header-container {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            padding: 3.5rem 2rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 25px rgba(30, 58, 138, 0.2);
            text-align: center;
            border-radius: 30px;
        }
        
        .header-container h1 {
            color: white;
            font-size: 2.8rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 0.75rem;
        }
        
        .header-container p {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
            font-weight: 400;
            max-width: 700px;
            margin: 0 auto;
        }
        
        .main-container {
            background-color: white;
            border-radius: 12px;
            padding: 2.5rem;
            box-shadow: 0 5px 20px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 2rem;
        }
        
        .stButton>button {
            background: linear-gradient(to right, #1e40af, #2563eb);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.85rem 2rem;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.2s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            background: linear-gradient(to right, #1e3a8a, #1d4ed8);
            box-shadow: 0 5px 15px rgba(30, 64, 175, 0.3);
        }
        
        .stNumberInput input, .stSelectbox select, .stTextInput input {
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            padding: 10px 14px !important;
        }
        
        .stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        }
        
        .result-container {
            background-color: #f8fafc;
            border-radius: 10px;
            padding: 2rem;
            margin-top: 2rem;
        }
        
        .progress-header {
            height: 6px;
            width: 100%;
            background-color: #e2e8f0;
            border-radius: 3px;
            margin: 1rem 0 2rem 0;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 1s ease;
        }
        
        .form-title {
            color: #1e293b;
            font-weight: 600;
            margin-bottom: 1.5rem;
            font-size: 1.4rem;
        }
        
        .footer {
            text-align: center;
            color: #64748b;
            font-size: 0.9rem;
            margin-top: 3rem;
        }
        
        .github-link {
            display: inline-block;
            margin-top: 1rem;
            color: #3b82f6;
            text-decoration: none;
            font-weight: 500;
        }
        
        .github-link:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-container">
        <h1>Placement Prediction System</h1>
        <p>Advanced machine learning model to predict candidate placement probability</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

with st.form("placement_form"):
    st.markdown('<div class="form-title">Candidate Information</div>', unsafe_allow_html=True)
    
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
        specialisation = st.radio("MBA Specialisation", ["Marketing and HR", "Marketing and Finance"])
        mba_p = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

    submit_button = st.form_submit_button(label="Predict Placement Probability")

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
        'specialisation': 1 if specialisation == 'Marketing and HR' else 0,
        'mba_p': mba_p
    }, index=[0])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    
    progress_color = "#10b981" if pred == 1 else "#ef4444"
    
    st.markdown(f"""
        <div class="progress-header">
            <div class="progress-fill" style="width: {prob*100}%; background: {progress_color};"></div>
        </div>
    """, unsafe_allow_html=True)
    
    if pred == 1:
        st.markdown("""
            <h3 style='color: #10b981;'>High Placement Probability</h3>
            <p>The analysis indicates strong potential for successful placement.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <h3 style='color: #ef4444;'>Low Placement Probability</h3>
            <p>The analysis suggests areas for improvement to enhance placement chances.</p>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style='margin-top: 1rem;'>
            <p style='font-weight: 500; margin-bottom: 0.5rem;'>Placement Probability: {prob:.1%}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <div class="footer">
        <a href="https://github.com/wigjatin/Student---Placement--Prediction-" class="github-link" target="_blank">
            View on GitHub
        </a>
    </div>
""", unsafe_allow_html=True)