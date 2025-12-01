import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import _encoders
from sklearn.utils import _encode as encode_module

# ============================================================
# COMPREHENSIVE SKLEARN PATCHES
# ============================================================

# Patch 1: sklearn.utils._encode._encode
_original_sklearn_encode = encode_module._encode

def safe_sklearn_encode(values, *, uniques, check_unknown=True):
    """
    Replacement for sklearn's _encode that handles mixed types by converting to strings.
    """
    values = np.asarray(values)
    uniques = np.asarray(uniques)
    
    # Convert everything to strings
    if values.ndim == 1:
        values_str = np.array([str(v) for v in values], dtype=object)
    else:
        values_str = np.array([[str(v) for v in row] for row in values], dtype=object)
    
    uniques_str = np.array([str(v) for v in uniques.ravel()], dtype=object)
    
    if check_unknown:
        diff = np.setdiff1d(values_str.ravel(), uniques_str)
        if len(diff) > 0:
            raise ValueError(f"y contains previously unseen labels: {diff}")
    
    # Create encoding via dictionary lookup
    encoder_dict = {val: idx for idx, val in enumerate(uniques_str)}
    
    if values_str.ndim == 1:
        encoded = np.array([encoder_dict.get(v, -1) for v in values_str], dtype=int)
    else:
        encoded = np.array([[encoder_dict.get(v, -1) for v in row] for row in values_str], dtype=int)
    
    return encoded

encode_module._encode = safe_sklearn_encode

# Patch 2: sklearn.utils._encode._check_unknown
_original_check_unknown = encode_module._check_unknown

def safe_check_unknown(values, known_values, return_mask=False):
    """
    Replacement for _check_unknown that handles mixed types.
    """
    values = np.asarray(values, dtype=object)
    known_values = np.asarray(known_values, dtype=object)
    
    # Convert to strings
    values_str = np.array([str(v) for v in values.ravel()], dtype=object).reshape(values.shape)
    known_str = np.array([str(v) for v in known_values.ravel()], dtype=object)
    
    if values_str.ndim == 1:
        mask = np.isin(values_str, known_str)
    else:
        mask = np.array([[np.isin(v, known_str) for v in row] for row in values_str])
    
    diff = values_str[~mask]
    unique_diff = np.unique(diff)
    
    if return_mask:
        return unique_diff, mask
    return unique_diff

encode_module._check_unknown = safe_check_unknown

# Patch 3: Override OneHotEncoder._transform method
_original_ohe_transform = OneHotEncoder._transform

def safe_ohe_transform(self, X, handle_unknown="error", force_all_finite=True, warn_on_unknown=False):
    """
    Patched OneHotEncoder._transform that converts inputs to strings first.
    """
    # Convert X to DataFrame if it isn't already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Convert all columns to strings
    X_str = X.copy()
    for col in X_str.columns:
        X_str[col] = X_str[col].astype(str)
    
    # Call original transform with string data
    return _original_ohe_transform(self, X_str, handle_unknown=handle_unknown, 
                                   force_all_finite=force_all_finite, 
                                   warn_on_unknown=warn_on_unknown)

OneHotEncoder._transform = safe_ohe_transform

# ============================================================
# Load and prepare model
# ============================================================
MODEL_PATH = "model_perm_best.pkl"

@st.cache_resource
def load_model():
    """Load and prepare the model with string-converted categories."""
    try:
        model = joblib.load(MODEL_PATH)
        
        def fix_ohe_categories(estimator):
            """Convert all OneHotEncoder categories to strings."""
            if isinstance(estimator, OneHotEncoder):
                if hasattr(estimator, 'categories_') and estimator.categories_ is not None:
                    estimator.categories_ = [
                        np.array([str(cat) for cat in cats], dtype=object)
                        for cats in estimator.categories_
                    ]
            elif isinstance(estimator, Pipeline):
                for _, step in estimator.steps:
                    fix_ohe_categories(step)
            elif isinstance(estimator, ColumnTransformer):
                for _, trans, _ in estimator.transformers_:
                    if trans not in ('drop', 'passthrough'):
                        fix_ohe_categories(trans)
            elif hasattr(estimator, 'estimators_'):
                for sub in estimator.estimators_:
                    fix_ohe_categories(sub)
        
        fix_ohe_categories(model)
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

model = load_model()

# ============================================================
# Streamlit UI
# ============================================================
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

# ============================================================
# Helper functions
# ============================================================
def to_annual(amount: float, unit: str) -> float:
    """Convert wage amount to annual equivalent."""
    multipliers = {"Year": 1.0, "Month": 12.0, "Week": 52.0, "Hour": 2080.0}
    return amount * multipliers.get(unit, 1.0)

# ============================================================
# Prediction
# ============================================================
if st.button("Estimate Approval Probability"):
    try:
        # Process inputs
        pw_wage_val = float(pw_wage)
        offer_from_val = float(offer_from)
        offer_to_val = float(offer_to)

        # Calculate features
        pw_annual = to_annual(pw_wage_val, pw_unit)
        offer_mid = (offer_from_val + offer_to_val) / 2.0
        offer_annual = to_annual(offer_mid, offer_unit)
        wage_ratio = offer_annual / pw_annual if pw_annual > 0 else 0.0

        # Build input - all categorical as strings
        input_data = {
            "PW_WAGE": [pw_wage_val],
            "PW_UNIT_OF_PAY": [str(pw_unit)],
            "WAGE_OFFER_FROM": [offer_from_val],
            "WAGE_OFFER_TO": [offer_to_val],
            "WAGE_OFFER_UNIT_OF_PAY": [str(offer_unit)],
            "PW_WAGE_ANNUAL": [pw_annual],
            "OFFER_WAGE_ANNUAL": [offer_annual],
            "WAGE_RATIO": [wage_ratio],
            "PW_SOC_CODE": [str(soc_code)],
            "NAICS_CODE": [str(naics_code)],
            "MINIMUM_EDUCATION": [str(edu)],
            "WORKSITE_STATE": [str(state).upper()],
            "FW_OWNERSHIP_INTEREST": [str(ownership)],
            "FISCAL_YEAR": [2024],
        }
        
        df = pd.DataFrame(input_data)
        
        # Ensure strings
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)

        # Predict
        prob = float(model.predict_proba(df)[:, 1][0] * 100.0)

        # Display
        st.subheader("Estimated Approval Probability")
        st.markdown(f"### ✅ **{prob:.1f}%** likelihood of certification")
        st.caption(
            "Note: This estimate is based on FY2024 PERM disclosure patterns only and "
            "does not replace legal or case-specific advice."
        )

        with st.expander("Show model input features"):
            st.dataframe(df)

    except Exception as e:
        st.error("The model encountered an error while scoring this case.")
        st.write(f"**Internal error:** {e}")
        
        with st.expander("Debug Information"):
            import traceback
            st.code(traceback.format_exc())
