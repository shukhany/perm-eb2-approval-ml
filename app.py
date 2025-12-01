import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------------------------------------------------
# 1) Monkey-patch sklearn's _check_unknown to avoid np.isnan bug
# ------------------------------------------------------------
from sklearn.utils import _encode as _enc


def _check_unknown_safe(X, known_values, return_mask=False):
    """
    Replacement for sklearn.utils._encode._check_unknown that does NOT call
    np.isnan on mixed string/NaN arrays (which causes the ufunc 'isnan' error).

    Behavior: treat all categories as valid; if return_mask=True, mark values
    that are in known_values. This is sufficient for OneHotEncoder with
    handle_unknown='ignore' in your pipeline.
    """
    X = np.asarray(X)
    known_values = np.asarray(known_values)

    if known_values.size == 0:
        if return_mask:
            return np.array([], dtype=X.dtype), np.zeros(X.shape, dtype=bool)
        return np.array([], dtype=X.dtype)

    # Boolean mask: which inputs are in the known categories?
    mask = np.isin(X, known_values)

    # "diff" = values not in known_values (we don't actually use it downstream,
    # but sklearn expects a set/array back)
    diff = X[~mask]
    unique_diff = np.unique(diff)

    if return_mask:
        return unique_diff, mask
    else:
        return unique_diff


# Patch it BEFORE loading the model
_enc._check_unknown = _check_unknown_safe

# ------------------------------------------------------------
# 2) Load trained pipeline
# ------------------------------------------------------------
MODEL_PATH = "model_perm_best.pkl"
model = joblib.load(MODEL_PATH)

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
    edu = st.selectbox("Minimum Education Required", ["High School", "Bachelor's", "Master's", "PhD"], index=2)
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
# 4) Helper to annualize wages (must match training logic)
# ------------------------------------------------------------
def to_annual(amount: float, unit: str) -> float:
    if unit == "Year":
        return amount
    if unit == "Month":
        return amount * 12.0
    if unit == "Week":
        return amount * 52.0
    if unit == "Hour":
        # standard 40h * 52w = 2,080h
        return amount * 2080.0
    # Fallback: no conversion
    return amount


# ------------------------------------------------------------
# 5) Build model input row & predict
# ------------------------------------------------------------
if st.button("Estimate Approval Probability"):
    try:
        # Clean numeric inputs
        pw_wage_val = float(pw_wage)
        offer_from_val = float(offer_from)
        offer_to_val = float(offer_to)

        # Annualize
        pw_annual = to_annual(pw_wage_val, pw_unit)
        offer_mid = (offer_from_val + offer_to_val) / 2.0
        offer_annual = to_annual(offer_mid, offer_unit)

        # Wage ratio (offered / prevailing)
        wage_ratio = offer_annual / pw_annual if pw_annual > 0 else np.nan

        # Build single-row DataFrame in EXACT schema used in training
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
            "MINIMUM_EDUCATION": [edu],
            "WORKSITE_STATE": [state],
            "FW_OWNERSHIP_INTEREST": [ownership],
            "FISCAL_YEAR": [2024],
        }

        df = pd.DataFrame(input_data)

        # Predict probability of approval
        prob = model.predict_proba(df)[:, 1][0] * 100.0
        prob = float(prob)

        st.subheader("Estimated Approval Probability")
        st.markdown(f"### ✅ **{prob:.1f}%** likelihood of certification (model estimate)")

        st.caption(
            "Note: This estimate is based on FY2024 PERM disclosure patterns only and "
            "does not replace legal or case-specific advice."
        )

        # Optional: show engineered features for debugging / explanation
        with st.expander("Show model input features"):
            st.dataframe(df)

    except Exception as e:
        st.error("The model encountered an error while scoring this case.")
        # For debugging during development; logs will capture full stack trace
        st.write(f"**Internal error:** {e}")
