import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    return joblib.load("model_perm_best.pkl")

model = load_model()

st.title("EB-2 PERM Approval Probability Estimator (FY2024)")

st.markdown("This tool estimates the probability that a PERM case will be certified using FY2024 data.")

col1, col2 = st.columns(2)

with col1:
    pw_soc_code = st.text_input("Prevailing Wage SOC Code", "15-1252")
    naics_code = st.text_input("Employer NAICS Code", "5415")
    minimum_education = st.selectbox(
        "Minimum Education Required",
        ["High School", "Associate", "Bachelor's", "Master's", "Doctorate"],
        index=2
    )
    worksite_state = st.text_input("Worksite State", "AZ")

with col2:
    pw_wage = st.number_input("Prevailing Wage Amount", min_value=0.0, value=90000.0)
    pw_unit = st.selectbox("Prevailing Wage Unit", ["Year", "Month", "Week", "Bi-Weekly", "Hour"])
    wage_offer_from = st.number_input("Wage Offer From", min_value=0.0, value=95000.0)
    wage_offer_to   = st.number_input("Wage Offer To", min_value=0.0, value=105000.0)
    wage_offer_unit = st.selectbox("Wage Offer Unit", ["Year", "Month", "Week", "Bi-Weekly", "Hour"])

ownership_interest = st.selectbox(
    "Ownership Interest?",
    ["N", "Y"],
    index=0
)

if st.button("Estimate Approval Probability"):

    input_row = {
        "PW_SOC_CODE": [pw_soc_code],
        "NAICS_CODE": [naics_code],
        "PW_WAGE": [pw_wage],
        "PW_UNIT_OF_PAY": [pw_unit],
        "WAGE_OFFER_FROM": [wage_offer_from],
        "WAGE_OFFER_TO": [wage_offer_to],
        "WAGE_OFFER_UNIT_OF_PAY": [wage_offer_unit],
        "MINIMUM_EDUCATION": [minimum_education],
        "WORKSITE_STATE": [worksite_state],
        "FW_OWNERSHIP_INTEREST": [ownership_interest],
        "FISCAL_YEAR": [2024],
    }

    df = pd.DataFrame(input_row)
    prob = model.predict_proba(df)[:, 1][0] * 100

    st.subheader(f"Approval Probability: {prob:.1f}%")
