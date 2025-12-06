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
    <p style="font-size: 1.1rem; margin: 0.5rem 0;">Predict approval probability for PERM labor certification</p>
    <p style="font-size: 0.95rem; margin: 1rem 2rem; line-height: 1.6; opacity: 0.95;">
        This machine learning model analyzes historical PERM data (2022-2024) to predict approval probability
        for EB-2 labor certification applications. The model evaluates wage ratios, education requirements,
        geographic factors, and occupation codes to provide data-driven insights for case assessment.
    </p>
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
        
        # Calculate key factors
        actual_offer = (offer_wage_from + (offer_wage_to if offer_wage_to > 0 else offer_wage_from)) / 2
        wage_ratio = actual_offer / pw_wage if pw_wage > 0 else 1
        wage_premium = actual_offer - pw_wage
        
        # Analyze factors affecting the score
        st.subheader("🔍 What's Affecting Your Score?")
        
        positive_factors = []
        negative_factors = []
        neutral_factors = []
        
        # Wage analysis
        if wage_ratio >= 1.2:
            positive_factors.append(f"✅ **Strong wage offer** - Offering {wage_ratio:.1%} of prevailing wage (${actual_offer:,.0f} vs ${pw_wage:,.0f})")
        elif wage_ratio >= 1.0:
            neutral_factors.append(f"➖ **Adequate wage offer** - Offering {wage_ratio:.1%} of prevailing wage (${actual_offer:,.0f} vs ${pw_wage:,.0f})")
        else:
            negative_factors.append(f"❌ **Below prevailing wage** - Offering only {wage_ratio:.1%} of prevailing wage (${actual_offer:,.0f} vs ${pw_wage:,.0f}). This significantly reduces approval chances.")
        
        # Education analysis
        education_level = EDUCATION_MAP.get(education, 2)
        if education_level >= 4:  # Masters or higher
            positive_factors.append(f"✅ **Advanced education required** - {education} positions typically have higher approval rates")
        elif education_level >= 3:  # Bachelors
            neutral_factors.append(f"➖ **Standard education requirement** - {education} is common for EB-2 applications")
        else:
            negative_factors.append(f"❌ **Lower education requirement** - {education} may not meet typical EB-2 standards")
        
        # Geographic analysis
        region = STATE_REGIONS.get(state, 'Unknown')
        high_demand_states = ['CA', 'NY', 'TX', 'WA', 'MA', 'VA', 'NJ']
        if state in high_demand_states:
            positive_factors.append(f"✅ **High-demand state** - {state} ({region}) has strong tech/professional job markets")
        else:
            neutral_factors.append(f"➖ **Location: {state}** - {region} region (approval rates vary by state)")
        
        # Ownership analysis
        if ownership == "Yes":
            negative_factors.append("❌ **Ownership interest** - Foreign worker ownership can complicate PERM applications and reduce approval probability")
        else:
            positive_factors.append("✅ **No ownership interest** - Clean employer-employee relationship")
        
        # SOC Code analysis
        soc_major = str(soc_code)[:2]
        if soc_major in ['15', '11', '13', '17', '19']:  # Tech, management, business, engineering, science
            positive_factors.append(f"✅ **Strong occupation category** - SOC {soc_code} typically has good approval rates")
        else:
            neutral_factors.append(f"➖ **Occupation: SOC {soc_code}** - Approval rates vary by specific occupation")
        
        # Display factors in columns
        col1, col2 = st.columns(2)
        
        with col1:
            if positive_factors:
                st.markdown("**✅ Positive Factors**")
                for factor in positive_factors:
                    st.markdown(factor)
            if len(positive_factors) == 0:
                st.markdown("**✅ Positive Factors**")
                st.markdown("*No significant positive factors identified*")
        
        with col2:
            if negative_factors:
                st.markdown("**❌ Negative Factors**")
                for factor in negative_factors:
                    st.markdown(factor)
            if len(negative_factors) == 0:
                st.markdown("**❌ Negative Factors**")
                st.markdown("*No significant negative factors identified*")
        
        if neutral_factors:
            st.markdown("**➖ Neutral Factors**")
            for factor in neutral_factors:
                st.markdown(factor)
        
        # Improvement suggestions
        if approval_probability < 80:
            st.markdown("---")
            st.subheader("💡 How to Improve Your Score")
            suggestions = []
            
            if wage_ratio < 1.15:
                increase_needed = pw_wage * 1.2 - actual_offer
                suggestions.append(f"**Increase offered wage** - Consider offering at least ${pw_wage * 1.2:,.0f} (20% above prevailing wage, +${increase_needed:,.0f})")
            
            if education_level < 4 and soc_major in ['15', '11', '13']:
                suggestions.append("**Strengthen education requirements** - If possible, require a Master's degree to better align with EB-2 standards")
            
            if ownership == "Yes":
                suggestions.append("**Address ownership concerns** - Consult with immigration attorney about ownership structure and its impact on PERM")
            
            suggestions.append("**Verify job requirements** - Ensure all requirements are truly necessary for the position and match actual business needs")
            suggestions.append("**Review recruitment documentation** - Strong recruitment efforts can support your case")
            
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"{i}. {suggestion}")
        
        # Additional insights
        with st.expander("📊 View Input Summary"):
            st.write("**Wage Information:**")
            st.write(f"- Prevailing Wage: ${pw_wage:,}")
            st.write(f"- Offered Wage: ${offer_wage_from:,}" + (f" - ${offer_wage_to:,}" if offer_wage_to > 0 else ""))
            st.write(f"- Wage Ratio: {wage_ratio:.2f} ({wage_ratio:.1%} of prevailing)")
            st.write(f"- Wage Premium: ${wage_premium:,.0f}")
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
