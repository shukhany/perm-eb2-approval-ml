import streamlit as st
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ------------------------------------------------------------
# 1) Load trained pipeline
# ------------------------------------------------------------
MODEL_PATH = "model_perm_best.pkl"
model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------
# 2) Fix OneHotEncoder categories: make EVERYTHING strings
#    to avoid mixed-type comparisons / np.isnan issues
# ------------------------------------------------------------
def _fix_ohe_categories(estimator):
    """Recursively walk the estimator and make all OneHotEncoder
    categories_ arrays pure strings (dtype=object)."""
    if isinstance(estimator, OneHotEncoder):
        if hasattr(estimator, "categories_") and estimator.categories_ is not None:
            new_cats = []
            for arr in estimator.categories_:
                arr = np.asarray(arr, dtype=object)
                # cast every entry to string, including nan -> 'nan'
                new_cats.append(np.array([str(x) for x in arr], dtype=object))
            estimator.categories_ = new_cats

    elif isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            _fix_ohe_categories(step)

    elif isinstance(estimator, ColumnTransformer):
        for _, trans, _ in estimator.transformers:
            if trans in ("drop", "passthrough"):
                continue
            _fix_ohe_categories(trans)

    # for safety: handle ensembles with sub-estimators_
    elif hasattr(estimator, "estimators_"):
        for sub in estimator.estimators_:
            _fix_ohe_categories(sub)


_fix_ohe_categories(model)

# ------------------------------------------------------------
# 3) Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="EB-2 PERM Approval Probability (FY2024)", layout="centered")

st.title("EB-2 PERM Approval Probability Estimator (FY2024)")

st.write(
    "This tool estimates the **probability that a PERM case will be certified** "
    "based on patterns in FY2024 PERM disclosure data. "
    "It is for **exploratory, educational use only** and does **not** constitute legal advice."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    soc_code = st.text_input("Prevailing Wage SOC Code", value="15-1252")
with col2:
    naics_code = st.text_input("Employer NAICS Code", value="5415")

with col1:
    pw_wage = st.number_input("Prevailing Wage Amount", min_value=0.0, value=115000.0, step=1000.0)
with col2:
    pw_unit = st.selectbox("Prevailing Wage Unit", ["Year", "Month", "Week", "Hour"], index=0)

with col1:
    edu = st.selectbox(
        "Minimum Education Required",
        ["High School", "Bachelor's", "Master's", "PhD"],
        index=2,
    )
with col2:
    state = st.text_input("Worksite State (e.g. CA, TX)", value="CA")

with col1:
    offer_from = st.number_input("Wage Offer From", min_value=0.0, value=130000.0, step=1000.0)
with col2:
    offer_to = st.number_input("Wage Offer To", min_value=0.0, value=140000.0, step=1000.0)

with col1:
    offer_unit = st.selectbox("Wage Offer Unit", ["Year", "Month", "Week", "Hour"], index=0)
with col2:
    ownership = st.selectbox("Ownership Interest?", ["N", "Y"], index=0)

st.markdown("---")

# ------------------------------------------------------------
# 4) Helper to annualize wages (same logic as training)
# ------------------------------------------------------------
def to_annual(amount: float, unit: str) -> float:
    if unit == "Year":
        return amount
    if unit == "Month":
        return amount * 12.0
    if unit == "Week":
        return amount * 52.0
    if unit == "Hour":
        # standard 40h/week * 52 weeks
        return amount * 2080.0
    return amount


# ------------------------------------------------------------
# 5) Build model input row & predict
# ------------------------------------------------------------
if st.button("Estimate Approval Probability"):
    try:
        pw_wage_val = float(pw_wage)
        offer_from_val = float(offer_from)
        offer_to_val = float(offer_to)

        pw_annual = to_annual(pw_wage_val, pw_unit)
        offer_mid = (offer_from_val + offer_to_val) / 2.0
        offer_annual = to_annual(offer_mid, offer_unit)
        wage_ratio = offer_annual / pw_annual if pw_annual > 0 else np.nan

        # Build one-row DataFrame in the exact schema used at training time
        input_data = {
            "PW_WAGE": [pw_wage_val],
            "PW_UNIT_OF_PAY": [pw_unit],
            "WAGE_OFFER_FROM": [offer_from_val],
            "WAGE_OFFER_TO": [offer_to_val],
            "WAGE_OFFER_UNIT_OF_PAY": [offer_unit],
            "PW_WAGE_ANNUAL": [pw_annual],
            "OFFER_WAGE_ANNUAL": [offer_annual],
            "WAGE_RATIO": [wage_ratio],
            "PW_SOC_CODE": [str(soc_code)],
            "NAICS_CODE": [str(naics_code)],
            "MINIMUM_EDUCATION": [str(edu)],
            "WORKSITE_STATE": [str(state)],
            "FW_OWNERSHIP_INTEREST": [str(ownership)],
            "FISCAL_YEAR": [2024],
        }

        df = pd.DataFrame(input_data)

        prob = model.predict_proba(df)[:, 1][0] * 100.0
        prob = float(prob)

        st.subheader("Estimated Approval Probability")
        st.markdown(f"### ✅ **{prob:.1f}%** likelihood of certification (model estimate)")

        st.caption(
            "Note: This estimate is based on FY2024 PERM disclosure patterns only and "
            "does not replace legal or case-specific advice."
        )

        with st.expander("Show model input features"):
            st.dataframe(df)

    except Exception as e:
        st.error("The model encountered an error while scoring this case.")
        st.write(f"**Internal error:** {e}")
