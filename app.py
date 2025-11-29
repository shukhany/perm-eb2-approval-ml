import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------- Load trained pipeline --------
@st.cache_resource
def load_model():
    return joblib.load("model_perm_best.pkl")

model = load_model()

st.title("EB-2 PERM Approval Probability Estimator (FY2024)")
st.markdown(
    "This tool estimates the probability that a PERM case will be certified using FY2024 data."
)

# -------- Helper: convert wage + unit to annual --------
WAGE_FACTORS = {
    "Year": 1.0,
    "Month": 12.0,
    "Week": 52.0,
    "Bi-Weekly": 26.0,
    "Hour": 2080.0,
}

def to_annual(wage, unit):
    factor = WAGE_FACTORS.get(unit, 1.0)
    return float(wage) * factor


# -------- UI Inputs --------
col1, col2 = st.columns(2)

with col1:
    pw_soc_code = st.text_input("Prevailing Wage SOC Code", "15-1252")
    naics_code = st.text_input("Employer NAICS Code", "5415")
    minimum_education = st.selectbox(
        "Minimum Education Required",
        ["High School", "Associate", "Bachelor's", "Master's", "Doctorate"],
        index=3,
    )
    worksite_state = st.text_input("Worksite State", "AZ")

with col2:
    pw_wage = st.number_input(
        "Prevailing Wage Amount", min_value=0.0, value=90000.0, step=1000.0
    )
    pw_unit = st.selectbox(
        "Prevailing Wage Unit", ["Year", "Month", "Week", "Bi-Weekly", "Hour"], index=0
    )

    wage_offer_from = st.number_input(
        "Wage Offer From", min_value=0.0, value=95000.0, step=1000.0
    )
    wage_offer_to = st.number_input(
        "Wage Offer To", min_value=0.0, value=105000.0, step=1000.0
    )
    wage_offer_unit = st.selectbox(
        "Wage Offer Unit", ["Year", "Month", "Week", "Bi-Weekly", "Hour"], index=0
    )

ownership_interest = st.selectbox("Ownership Interest?", ["N", "Y"], index=0)

st.markdown("---")

if st.button("Estimate Approval Probability"):

    # ---- Recreate engineered features used in training ----
    # 1. Annualized prevailing wage
    pw_wage_annual = to_annual(pw_wage, pw_unit)

    # 2. Annualized offered wage (use midpoint of range)
    offer_mid = (wage_offer_from + wage_offer_to) / 2.0
    offer_wage_annual = to_annual(offer_mid, wage_offer_unit)

    # 3. Wage ratio (offer / prevailing) with safe divide
    if pw_wage_annual > 0:
        wage_ratio = offer_wage_annual / pw_wage_annual
    else:
        wage_ratio = 1.0  # fallback to neutral ratio

    # ---- Build DataFrame with EXACT training columns ----
    input_data = {
        "PW_WAGE_ANNUAL": [pw_wage_annual],
        "OFFER_WAGE_ANNUAL": [offer_wage_annual],
        "WAGE_RATIO": [wage_ratio],
        "PW_SOC_CODE": [pw_soc_code],
        "NAICS_CODE": [naics_code],
        "MINIMUM_EDUCATION": [minimum_education],
        "WORKSITE_STATE": [worksite_state],
        "FW_OWNERSHIP_INTEREST": [ownership_interest],
        "FISCAL_YEAR": [2024],
    }

    df = pd.DataFrame(input_data)

    # ---- Predict probability ----
    prob = model.predict_proba(df)[:, 1][0] * 100.0
    prob = float(prob)

    st.subheader("Estimated Approval Probability")
    st.metric("Approval Probability", f"{prob:.1f}%")

    # Simple risk band
    if prob >= 85:
        risk = "Low risk"
    elif prob >= 65:
        risk = "Moderate risk"
    else:
        risk = "High risk"

    st.markdown(f"**Risk Category:** {risk}")
    st.caption(
        "Note: This is a statistical estimate based on FY2024 PERM outcomes and does not constitute legal advice."
    )
