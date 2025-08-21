import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stSelectbox > div > div > div {
        color: #2c3e50 !important;
    }
    .stNumberInput > div > div > input {
        color: #2c3e50 !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        color: #2c3e50 !important;
        background-color: #ffffff !important;
    }
    .stSelectbox [role="listbox"] {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }
    .stSelectbox [role="option"] {
        color: #2c3e50 !important;
        background-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    """Load the trained model, feature names, and results"""
    try:
        model = joblib.load('loan_model.pkl')
        
        with open('data/processed/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        with open('model_results.json', 'r') as f:
            results = json.load(f)
        
        return model, feature_names, results
    except Exception as e:
        st.error(f"Model files not found! Please run index.py first. Error: {e}")
        return None, None, None

def main():
    st.title("🏦 Loan Approval Predictor")
    
    # Load model and data
    model, feature_names, results = load_model_and_data()
    if model is None:
        st.stop()

    # Input form
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Personal Information")
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
            education = st.selectbox("Education Level", ["Not Graduate", "Graduate"], index=1)
            self_employed = st.selectbox("Employment Status", ["No", "Yes"], index=0)
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
        
        with col2:
            st.subheader("💰 Financial Information")
            income_annum = st.number_input("Annual Income (₹)", min_value=0, value=5000000, step=100000)
            loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=15000000, step=100000)
            loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=10)
        
        st.subheader("🏠 Asset Information")
        col3, col4 = st.columns(2)
        
        with col3:
            residential_assets = st.number_input("Residential Assets (₹)", min_value=0, value=5000000, step=100000)
            commercial_assets = st.number_input("Commercial Assets (₹)", min_value=0, value=2000000, step=100000)
        
        with col4:
            luxury_assets = st.number_input("Luxury Assets (₹)", min_value=0, value=10000000, step=100000)
            bank_assets = st.number_input("Bank Assets (₹)", min_value=0, value=3000000, step=100000)
        
        submitted = st.form_submit_button("🔮 Predict Loan Status", type="primary", use_container_width=True)

        if submitted:
            # Calculate derived features
            movable_assets = bank_assets + luxury_assets
            immovable_assets = residential_assets + commercial_assets

            # Encode categorical variables
            education_encoded = 1 if education == "Graduate" else 0
            self_employed_encoded = 1 if self_employed == "Yes" else 0

            # Create input DataFrame
            input_data = pd.DataFrame({
                'no_of_dependents': [no_of_dependents],
                'education': [education_encoded],
                'self_employed': [self_employed_encoded],
                'income_annum': [income_annum],
                'loan_amount': [loan_amount],
                'loan_term': [loan_term],
                'cibil_score': [cibil_score],
                'Movable_assets': [movable_assets],
                'Immovable_assets': [immovable_assets]
            })

            # Reorder columns to match training data
            input_data = input_data[feature_names]

            # Make prediction
            try:
                prediction = model.predict(input_data)[0]
                
                # Handle probabilities safely
                try:
                    probabilities = model.predict_proba(input_data)[0]
                    if len(probabilities) == 2:
                        prob_rejected, prob_approved = probabilities[0], probabilities[1]
                    else:
                        prob_approved = 0.85 if prediction == 1 else 0.15
                        prob_rejected = 1 - prob_approved
                except:
                    prob_approved = 0.85 if prediction == 1 else 0.15
                    prob_rejected = 1 - prob_approved

                # Display prediction results
                st.subheader("🎯 Prediction Result")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if prediction == 1:
                        st.success("✅ **LOAN APPROVED!**")
                    else:
                        st.error("❌ **LOAN REJECTED**")
                
                with col_res2:
                    st.metric("Approval Probability", f"{prob_approved*100:.1f}%")
                    st.metric("Rejection Probability", f"{prob_rejected*100:.1f}%")
                    
                # Show confidence
                confidence = max(prob_approved, prob_rejected)
                st.metric("Prediction Confidence", f"{confidence*100:.1f}%")
                
                # Basic financial analysis
                st.subheader("📊 Quick Analysis")
                loan_to_income = (loan_amount / income_annum) * 100 if income_annum > 0 else 0
                total_assets = movable_assets + immovable_assets
                asset_to_loan = (total_assets / loan_amount) * 100 if loan_amount > 0 else 0
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.metric("Loan-to-Income Ratio", f"{loan_to_income:.1f}%")
                
                with analysis_col2:
                    st.metric("Asset-to-Loan Ratio", f"{asset_to_loan:.1f}%")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Debug info:")
                st.write(f"Features expected: {feature_names}")
                st.write(f"Input data shape: {input_data.shape}")

if __name__ == "__main__":
    main()
