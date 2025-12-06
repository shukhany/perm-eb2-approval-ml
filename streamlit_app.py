"""
PERM EB2 Approval Prediction - Streamlit Web Application
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# =================================================================================
# PAGE CONFIG
# =================================================================================
st.set_page_config(
    page_title="PERM EB2 Approval Predictor",
    page_icon="🏛️",
    layout="centered"
)

# =================================================================================
# LOAD MODEL
# =================================================================================
MODEL_PATH = 'best_model_for_deployment.pkl'

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at {MODEL_PATH}")
        return None

model = load_model()

# =================================================================================
# CONFIGURATION
# =================================================================================
EDUCATION_MAP = {
    "None": 0, "High School": 1, "Associate's Degree": 2,
    "Bachelor's Degree": 3, "Master's Degree": 4, "Doctorate": 5
}

STATE_REGIONS = {
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
    'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast',
    'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest', 'WI': 'Midwest',
    'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest',
    'ND': 'Midwest', 'SD': 'Midwest',
    'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South',
    'SC': 'South', 'VA': 'South', 'DC': 'South', 'WV': 'South', 'AL': 'South',
    'KY': 'South', 'MS': 'South', 'TN': 'South', 'AR': 'South', 'LA': 'South',
    'OK': 'South', 'TX': 'South',
    'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
    'NM': 'West', 'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West',
    'HI': 'West', 'OR': 'West', 'WA': 'West'
}

US_STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'DC': 'District of Columbia', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii',
    'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine',
    'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska',
    'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
    'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island',
    'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas',
    'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

# =================================================================================
# HELPER FUNCTION
# =================================================================================
def preprocess_input(pw_wage, offer_wage_from, offer_wage_to, education, state, 
                     year, soc_code, naics_code, ownership):
    offer_wage_to = offer_wage_to or offer_wage_from
    offer_wage = (offer_wage_from + offer_wage_to) / 2
    wage_ratio = offer_wage / pw_wage if pw_wage > 0 else 1
    wage_premium = offer_wage - pw_wage
    
    # Map education to numeric
    education_level = EDUCATION_MAP.get(education, 2)
    
    return pd.DataFrame([{
        'PW_WAGE_ANNUAL': pw_wage,
        'OFFER_WAGE_ANNUAL': offer_wage,
        'WAGE_RATIO': wage_ratio,
        'WAGE_PREMIUM': wage_premium,
        'LOG_PW_WAGE': np.log1p(pw_wage),
        'LOG_OFFER_WAGE': np.log1p(offer_wage),
        'EDUCATION_LEVEL': education_level,
        'HAS_OWNERSHIP': 1 if ownership == 'Yes' else 0,
        'DATA_YEAR': year,
        'SOC_MAJOR': str(soc_code)[:2],
        'NAICS_SECTOR': str(naics_code)[:2],
        'REGION': STATE_REGIONS.get(state, 'West'),
        'WORKSITE_STATE': state
    }])

# =================================================================================
# CUSTOM CSS
# =================================================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    .prediction-box.green {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
    }
    .prediction-box.orange {
        background: linear-gradient(135deg, #fdcb6e, #e17055);
        color: white;
    }
    .prediction-box.red {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
    }
    .probability {
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .risk-level {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =================================================================================
# MAIN APP
# =================================================================================
st.markdown("""
<div class="main-header">
    <h1>🏛️ PERM EB2 Approval Predictor</h1>
    <p>Predict approval probability for PERM labor certification</p>
    <span style="background: rgba(255,255,255,0.2); padding: 0.25rem 1rem; border-radius: 20px; font-size: 0.85rem;">
        Model optimized for F1 Score
    </span>
</div>
""", unsafe_allow_html=True)

# Check if model is loaded
if model is None:
    st.error("⚠️ Model not loaded. Please ensure 'best_model_for_deployment.pkl' is in the app directory.")
    st.stop()

# =================================================================================
# INPUT FORM
# =================================================================================
with st.form("prediction_form"):
    st.subheader("📝 Application Details")
    
    # Wage Information
    col1, col2 = st.columns(2)
    with col1:
        pw_wage = st.number_input(
            "Prevailing Wage (Annual $)",
            min_value=0,
            value=100000,
            step=1000,
            help="DOL-determined prevailing wage for the position"
        )
    with col2:
        offer_wage_from = st.number_input(
            "Offered Wage From ($)",
            min_value=0,
            value=120000,
            step=1000,
            help="Minimum offered wage"
        )
    
    offer_wage_to = st.number_input(
        "Offered Wage To ($ - Optional)",
        min_value=0,
        value=0,
        step=1000,
        help="Maximum offered wage (leave 0 if same as 'From')"
    )
    
    # Education & Location
    col3, col4 = st.columns(2)
    with col3:
        education = st.selectbox(
            "Minimum Education Required",
            options=list(EDUCATION_MAP.keys()),
            index=3  # Default to Bachelor's
        )
    with col4:
        state = st.selectbox(
            "Worksite State",
            options=list(US_STATES.keys()),
            format_func=lambda x: f"{x} - {US_STATES[x]}",
            index=list(US_STATES.keys()).index('CA')  # Default to California
        )
    
    # Codes & Year
    col5, col6, col7 = st.columns(3)
    with col5:
        year = st.selectbox("Filing Year", [2024, 2025], index=0)
    with col6:
        soc_code = st.text_input("SOC Code", value="15-1252", help="e.g., 15-1252 (Software Developers)")
    with col7:
        naics_code = st.text_input("NAICS Code", value="54", help="e.g., 541511 (Custom Software)")
    
    # Ownership
    ownership = st.radio(
        "Foreign Worker Has Ownership Interest?",
        options=["No", "Yes"],
        index=0,
        horizontal=True
    )
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict Approval Probability", use_container_width=True)

# =================================================================================
# PREDICTION
# =================================================================================
if submitted:
    try:
        # Preprocess input
        input_df = preprocess_input(
            pw_wage, offer_wage_from, offer_wage_to, education, 
            state, year, soc_code, naics_code, ownership
        )
        
        # Make prediction
        proba = model.predict_proba(input_df)[0]
        approval_probability = round(proba[1] * 100, 1)
        
        # Determine risk level
        if approval_probability >= 80:
            risk_level = "LOW RISK"
            risk_color = "green"
            recommendation = "✅ Strong case. Standard processing recommended."
            icon = "🟢"
        elif approval_probability >= 60:
            risk_level = "MODERATE RISK"
            risk_color = "orange"
            recommendation = "⚠️ Review case details. Consider strengthening weak areas."
            icon = "🟡"
        else:
            risk_level = "HIGH RISK"
            risk_color = "red"
            recommendation = "🔴 Significant concerns. Detailed review recommended before filing."
            icon = "🔴"
        
        # Display results
        st.markdown(f"""
        <div class="prediction-box {risk_color}">
            <div style="font-size: 3rem;">{icon}</div>
            <div class="probability">{approval_probability}%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Approval Probability</div>
            <div class="risk-level">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"**Recommendation:** {recommendation}")
        
        # Additional insights
        with st.expander("📊 View Input Summary"):
            st.write("**Wage Information:**")
            st.write(f"- Prevailing Wage: ${pw_wage:,}")
            st.write(f"- Offered Wage: ${offer_wage_from:,}" + (f" - ${offer_wage_to:,}" if offer_wage_to > 0 else ""))
            st.write(f"- Wage Ratio: {(offer_wage_from/pw_wage if pw_wage > 0 else 0):.2f}")
            st.write("\n**Position Details:**")
            st.write(f"- Education: {education}")
            st.write(f"- State: {state} ({STATE_REGIONS.get(state, 'Unknown')} Region)")
            st.write(f"- SOC Code: {soc_code}")
            st.write(f"- NAICS Code: {naics_code}")
            st.write(f"- Ownership Interest: {ownership}")
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.exception(e)

# =================================================================================
# FOOTER
# =================================================================================
st.markdown("---")
st.info("""
**Note:** This prediction tool is based on historical PERM data (2022-2024) and uses machine learning models 
selected based on F1 Score optimization. Results should be used as one factor in case assessment, 
not as the sole decision-making criterion.
""")
