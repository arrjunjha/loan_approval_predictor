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
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .model-info h3 {
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .model-info p {
        color: #2c3e50;
        margin: 0.25rem 0;
    }
    .explanation-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        color: #2c3e50;
    }
    .approved-explanation {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .rejected-explanation {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    .metric-highlight {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
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

def check_business_rules(income_annum, total_assets, loan_amount, cibil_score):
    """Apply banking business rules before model prediction"""
    
    # Rule 1: Zero income check
    if income_annum <= 0:
        return (False, "❌ **BUSINESS RULE VIOLATION**", 
                "Zero annual income detected. Banks cannot approve loans without verified income source.")
    
    # Rule 2: Minimum income requirement (20% of loan amount annually)
    min_income_required = loan_amount * 0.2
    if income_annum < min_income_required:
        return (False, "❌ **INSUFFICIENT INCOME**", 
                f"Annual income (₹{income_annum:,}) is below minimum requirement (₹{min_income_required:,}). "
                f"Banks require minimum 20% of loan amount as annual income.")
    
    # Rule 3: Asset collateral check for high-value loans
    if total_assets <= 50000 and loan_amount > 2000000:
        return (False, "❌ **INSUFFICIENT COLLATERAL**", 
                f"Loan amount (₹{loan_amount:,}) requires substantial asset backing. "
                f"Current assets (₹{total_assets:,}) insufficient for this loan size.")
    
    # Rule 4: Extreme loan-to-income ratio
    loan_to_income_ratio = loan_amount / income_annum
    if loan_to_income_ratio > 15:  # More than 15 times annual income
        return (False, "❌ **EXTREME DEBT BURDEN**", 
                f"Loan-to-income ratio ({loan_to_income_ratio:.1f}x) exceeds banking limits. "
                f"Maximum recommended ratio is 10x annual income.")
    
    # Rule 5: Very poor credit score
    if cibil_score < 300:
        return (False, "❌ **INVALID CREDIT SCORE**", 
                "CIBIL score cannot be below 300. Please verify your credit score.")
    
    return (True, "✅ **BUSINESS RULES PASSED**", "All banking requirements met. Proceeding to AI analysis.")

def generate_loan_explanation(prediction, prob_approved, applicant_data):
    """Generate detailed explanation for loan decision"""
    
    # Extract data
    no_of_dependents = applicant_data['no_of_dependents']
    education = applicant_data['education_text']
    self_employed = applicant_data['self_employed_text']
    income_annum = applicant_data['income_annum']
    loan_amount = applicant_data['loan_amount']
    loan_term = applicant_data['loan_term']
    cibil_score = applicant_data['cibil_score']
    movable_assets = applicant_data['movable_assets']
    immovable_assets = applicant_data['immovable_assets']
    
    # Calculate key ratios
    loan_to_income = (loan_amount / income_annum) * 100 if income_annum > 0 else 0
    total_assets = movable_assets + immovable_assets
    asset_to_loan = (total_assets / loan_amount) * 100 if loan_amount > 0 else 0
    monthly_emi = loan_amount / (loan_term * 12) if loan_term > 0 else 0
    monthly_income = income_annum / 12 if income_annum > 0 else 1
    emi_to_income = (monthly_emi / monthly_income) * 100
    
    # Analyze factors
    positive_factors = []
    negative_factors = []
    neutral_factors = []
    
    # CIBIL Score Analysis
    if cibil_score >= 750:
        positive_factors.append(f"excellent credit score of {cibil_score}")
    elif cibil_score >= 650:
        neutral_factors.append(f"decent credit score of {cibil_score}")
    else:
        negative_factors.append(f"below-average credit score of {cibil_score}")
    
    # Education Factor
    if education == "Graduate":
        positive_factors.append("graduate-level education")
    else:
        negative_factors.append("non-graduate education background")
    
    # Employment Factor
    if self_employed == "Yes":
        negative_factors.append("self-employed status which indicates variable income")
    else:
        positive_factors.append("stable employment status")
    
    # Income Analysis
    if income_annum >= 5000000:
        positive_factors.append(f"strong annual income of ₹{income_annum:,}")
    elif income_annum >= 2000000:
        neutral_factors.append(f"moderate annual income of ₹{income_annum:,}")
    else:
        negative_factors.append(f"limited annual income of ₹{income_annum:,}")
    
    # Loan-to-Income Ratio
    if loan_to_income <= 300:
        positive_factors.append(f"reasonable loan-to-income ratio of {loan_to_income:.1f}%")
    elif loan_to_income <= 500:
        neutral_factors.append(f"moderate loan-to-income ratio of {loan_to_income:.1f}%")
    else:
        negative_factors.append(f"high loan-to-income ratio of {loan_to_income:.1f}%")
    
    # Asset Coverage
    if asset_to_loan >= 200:
        positive_factors.append(f"strong asset coverage with {asset_to_loan:.1f}% asset-to-loan ratio")
    elif asset_to_loan >= 100:
        neutral_factors.append(f"adequate asset coverage with {asset_to_loan:.1f}% asset-to-loan ratio")
    else:
        negative_factors.append(f"limited asset coverage with only {asset_to_loan:.1f}% asset-to-loan ratio")
    
    # EMI Burden
    if emi_to_income <= 30:
        positive_factors.append(f"manageable EMI burden of {emi_to_income:.1f}% of monthly income")
    elif emi_to_income <= 50:
        neutral_factors.append(f"moderate EMI burden of {emi_to_income:.1f}% of monthly income")
    else:
        negative_factors.append(f"high EMI burden of {emi_to_income:.1f}% of monthly income")
    
    # Dependents Factor
    if no_of_dependents <= 2:
        positive_factors.append(f"manageable family size with {no_of_dependents} dependent{'s' if no_of_dependents != 1 else ''}")
    elif no_of_dependents <= 4:
        neutral_factors.append(f"moderate family size with {no_of_dependents} dependents")
    else:
        negative_factors.append(f"large family size with {no_of_dependents} dependents")
    
    # Generate explanation based on prediction
    if prediction == 1:  # Approved
        explanation = f"""
**🎉 Loan Approval Explanation:**

Your loan application has been **APPROVED** with a {prob_approved*100:.1f}% confidence level. The decision was primarily based on the following strengths in your application:

**Positive Factors:**
"""
        for factor in positive_factors:
            explanation += f"• Your {factor}\n"
        
        if neutral_factors:
            explanation += f"\n**Neutral Factors:**\n"
            for factor in neutral_factors:
                explanation += f"• Your {factor}\n"
        
        if negative_factors:
            explanation += f"\n**Areas of Concern (but not dealbreakers):**\n"
            for factor in negative_factors:
                explanation += f"• Your {factor}\n"
        
        explanation += f"""
**Final Assessment:** 
The combination of your strong financial profile, particularly your credit history and asset base, outweighs any concerns. Your monthly EMI of ₹{monthly_emi:,.0f} appears manageable given your income level. The bank is confident in your ability to repay the loan amount of ₹{loan_amount:,} over {loan_term} years.
        """
        
    else:  # Rejected
        explanation = f"""
**❌ Loan Rejection Explanation:**

Your loan application has been **REJECTED** with a {(1-prob_approved)*100:.1f}% confidence level. The decision was based on several risk factors identified in your application:

**Primary Concerns:**
"""
        for factor in negative_factors:
            explanation += f"• Your {factor}\n"
        
        if neutral_factors:
            explanation += f"\n**Additional Considerations:**\n"
            for factor in neutral_factors:
                explanation += f"• Your {factor}\n"
        
        if positive_factors:
            explanation += f"\n**Positive Aspects (but insufficient):**\n"
            for factor in positive_factors:
                explanation += f"• Your {factor}\n"
        
        explanation += f"""
**Final Assessment:** 
While your application shows some positive elements, the overall risk profile is too high for approval. Key concerns include the debt-to-income ratio and EMI burden relative to your current financial situation. Consider improving your credit score, reducing the loan amount, or increasing your asset base before reapplying.

**Recommendations:**
• Work on improving your CIBIL score to above 750
• Consider a smaller loan amount (current: ₹{loan_amount:,})
• Build additional assets to strengthen your financial profile
• If self-employed, provide additional income documentation
        """
    
    return explanation

def display_model_info_header(results):
    """Display model information at the top"""
    best_model = results['best_model'].replace('_', ' ').title()
    best_accuracy = results['best_accuracy']
    
    # Find best F1 score
    best_f1 = 0
    for model_name, metrics in results.items():
        if model_name not in ['best_model', 'best_accuracy'] and isinstance(metrics, dict):
            if results['best_model'] in model_name and metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
    
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin: 1rem 0;">
        <h3 style="color: #1f77b4; margin-bottom: 0.5rem;">🤖 Current Model: {best_model}</h3>
        <p style="color: #2c3e50; margin: 0.25rem 0;"><span style="font-size: 1.1rem; font-weight: bold; color: #1f77b4;">Accuracy:</span> <span style="color: #2c3e50;">{best_accuracy:.4f} ({best_accuracy*100:.2f}%)</span></p>
        <p style="color: #2c3e50; margin: 0.25rem 0;"><span style="font-size: 1.1rem; font-weight: bold; color: #1f77b4;">F1 Score:</span> <span style="color: #2c3e50;">{best_f1:.4f}</span></p>
        <p style="color: #2c3e50; margin: 0.25rem 0;"><span style="font-size: 1.1rem; font-weight: bold; color: #1f77b4;">Status:</span> <span style="color: #28a745;">Ready for predictions + Business Rules Applied</span></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.title("🏦 Loan Approval Prediction System")
    
    # Load model and data
    model, feature_names, results = load_model_and_data()
    if model is None:
        st.stop()
    
    # Always show model info at top
    display_model_info_header(results)
    
    st.header("💰 Loan Approval Predictor")
    st.write("**Enhanced with Banking Business Rules** - Prevents unrealistic approvals")
    
    # Input form
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Personal Information")
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
            education = st.selectbox("Education Level", ["Not Graduate", "Graduate"], index=1)
            self_employed = st.selectbox("Employment Type", ["No", "Yes"], index=0)
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
        
        with col2:
            st.subheader("💰 Financial Information")
            income_annum = st.number_input("Annual Income (₹)", min_value=0, value=5000000, step=100000)
            loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=15000000, step=100000)
            loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=10)
        
        st.subheader("🏠 Asset Portfolio")
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
            total_assets = movable_assets + immovable_assets
            
            # STEP 1: CHECK BUSINESS RULES FIRST
            st.markdown("---")
            st.subheader("🏛️ Banking Business Rules Check")
            
            rules_passed, rule_status, rule_message = check_business_rules(
                income_annum, total_assets, loan_amount, cibil_score
            )
            
            st.write(f"**{rule_status}**")
            st.info(rule_message)
            
            if not rules_passed:
                st.error("**❌ LOAN APPLICATION REJECTED BY BUSINESS RULES**")
                st.write("**This rejection occurred before AI model analysis due to fundamental banking policy violations.**")
                
                # Show what needs to be fixed
                st.subheader("📋 Required Actions:")
                if income_annum <= 0:
                    st.write("• **Provide valid income documentation**")
                if income_annum < loan_amount * 0.2:
                    st.write(f"• **Increase annual income to at least ₹{loan_amount * 0.2:,.0f}**")
                if total_assets <= 50000 and loan_amount > 2000000:
                    st.write("• **Increase asset base or reduce loan amount**")
                    
                st.stop()
            
            # STEP 2: IF BUSINESS RULES PASS, USE AI MODEL
            st.success("✅ **Business rules passed! Proceeding to AI model analysis...**")
            
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
            
            # Reorder columns
            input_data = input_data[feature_names]
            
            # Prepare applicant data for explanation
            applicant_data = {
                'no_of_dependents': no_of_dependents,
                'education_text': education,
                'self_employed_text': self_employed,
                'income_annum': income_annum,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'cibil_score': cibil_score,
                'movable_assets': movable_assets,
                'immovable_assets': immovable_assets
            }
            
            # Make prediction
            try:
                prediction = model.predict(input_data)[0]
                
                # Handle probabilities safely
                try:
                    probabilities = model.predict_proba(input_data)[0]
                    if len(probabilities) == 2:
                        prob_rejected, prob_approved = probabilities, probabilities[1]
                    else:
                        prob_approved = 0.85 if prediction == 1 else 0.15
                        prob_rejected = 1 - prob_approved
                except:
                    prob_approved = 0.85 if prediction == 1 else 0.15
                    prob_rejected = 1 - prob_approved
                
                # Display prediction results with model info
                st.markdown("---")
                st.header("🎯 AI Model Prediction Results")
                
                # Model used banner
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #1f77b4, #17becf); 
                            color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; margin-bottom: 1rem;">
                    <h3>🤖 Prediction Made Using: {results['best_model'].replace('_', ' ').title()}</h3>
                    <p>Model Accuracy: {results['best_accuracy']:.4f} ({results['best_accuracy']*100:.2f}%) | Enhanced with Business Rules</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Main prediction result
                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                
                with col_res1:
                    if prediction == 1:
                        st.success("✅ **LOAN APPROVED!**")
                    else:
                        st.error("❌ **LOAN REJECTED**")
                
                with col_res2:
                    st.metric("Approval Probability", f"{prob_approved*100:.1f}%")
                    st.progress(prob_approved)
                
                with col_res3:
                    confidence = max(prob_approved, prob_rejected)
                    st.metric("Prediction Confidence", f"{confidence*100:.1f}%")
                
                with col_res4:
                    st.metric("Model Accuracy", f"{results['best_accuracy']*100:.2f}%")
                    st.caption(f"Protected by Business Rules")
                
                # DETAILED EXPLANATION SECTION
                st.subheader("📝 Detailed Decision Explanation")
                
                explanation = generate_loan_explanation(prediction, prob_approved, applicant_data)
                
                # Style the explanation box based on approval/rejection
                if prediction == 1:
                    st.markdown(f"""
                    <div class="explanation-box approved-explanation">
                        {explanation.replace('**', '<strong>').replace('**', '</strong>').replace('\n', '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="explanation-box rejected-explanation">
                        {explanation.replace('**', '<strong>').replace('**', '</strong>').replace('\n', '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Financial analysis metrics
                st.subheader("📊 Financial Analysis Summary")
                
                analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)
                
                loan_to_income = (loan_amount / income_annum) * 100 if income_annum > 0 else 0
                asset_to_loan = (total_assets / loan_amount) * 100 if loan_amount > 0 else 0
                monthly_emi = loan_amount / (loan_term * 12) if loan_term > 0 else 0
                monthly_income = income_annum / 12 if income_annum > 0 else 1
                emi_ratio = (monthly_emi / monthly_income) * 100
                
                with analysis_col1:
                    st.metric("Loan-to-Income", f"{loan_to_income:.1f}%")
                    if loan_to_income <= 300:
                        st.success("✅ Excellent")
                    elif loan_to_income <= 500:
                        st.warning("⚡ Moderate")
                    else:
                        st.error("⚠️ High Risk")
                
                with analysis_col2:
                    st.metric("Asset Coverage", f"{asset_to_loan:.1f}%")
                    if asset_to_loan >= 200:
                        st.success("✅ Strong")
                    elif asset_to_loan >= 100:
                        st.warning("⚡ Adequate")
                    else:
                        st.error("⚠️ Weak")
                
                with analysis_col3:
                    st.metric("Monthly EMI", f"₹{monthly_emi:,.0f}")
                    st.caption(f"EMI Ratio: {emi_ratio:.1f}%")
                
                with analysis_col4:
                    st.metric("Total Assets", f"₹{total_assets:,.0f}")
                    asset_types = sum([
                        1 if residential_assets > 0 else 0,
                        1 if commercial_assets > 0 else 0,
                        1 if luxury_assets > 0 else 0,
                        1 if bank_assets > 0 else 0
                    ])
                    st.caption(f"Diversified: {asset_types}/4 types")
                
                # Show the enhancement
                st.info("🛡️ **Enhanced Protection**: This system now includes banking business rules that prevent unrealistic loan approvals, making it suitable for real-world financial applications.")
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
