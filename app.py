import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# MONKEY PATCHES - Must be applied before loading the model
# ============================================================

# Store original methods
_original_ndarray_lt = np.ndarray.__lt__
_original_ndarray_le = np.ndarray.__le__
_original_ndarray_gt = np.ndarray.__gt__
_original_ndarray_ge = np.ndarray.__ge__
_original_searchsorted = np.ndarray.searchsorted

def safe_comparison_method(original_method):
    """Wrapper for comparison methods that handles mixed types."""
    def wrapper(self, other):
        try:
            return original_method(self, other)
        except TypeError as e:
            if "'<' not supported" in str(e) or "'>' not supported" in str(e):
                # Convert both to strings and compare
                self_str = np.array([str(x) for x in self.flat], dtype=object).reshape(self.shape)
                if isinstance(other, np.ndarray):
                    other_str = np.array([str(x) for x in other.flat], dtype=object).reshape(other.shape)
                else:
                    other_str = str(other)
                return original_method(self_str, other_str)
            raise
    return wrapper

def safe_searchsorted(self, v, side='left', sorter=None):
    """Safe searchsorted that converts to strings if needed."""
    try:
        return _original_searchsorted(self, v, side=side, sorter=sorter)
    except TypeError as e:
        if "'<' not supported" in str(e):
            # Convert to strings
            self_str = np.array([str(x) for x in self.flat], dtype=object).reshape(self.shape)
            if isinstance(v, np.ndarray):
                v_str = np.array([str(x) for x in v.flat], dtype=object).reshape(v.shape)
            else:
                v_str = str(v) if not isinstance(v, (list, tuple)) else np.array([str(x) for x in v], dtype=object)
            return _original_searchsorted(self_str, v_str, side=side, sorter=sorter)
        raise

# Apply patches
np.ndarray.__lt__ = safe_comparison_method(_original_ndarray_lt)
np.ndarray.__le__ = safe_comparison_method(_original_ndarray_le)
np.ndarray.__gt__ = safe_comparison_method(_original_ndarray_gt)
np.ndarray.__ge__ = safe_comparison_method(_original_ndarray_ge)
np.ndarray.searchsorted = safe_searchsorted

# ============================================================
# 1) Load trained pipeline
# ============================================================
MODEL_PATH = "model_perm_best.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# ============================================================
# 2) Convert all OneHotEncoder categories to strings
# ============================================================
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

# ============================================================
# 3) Streamlit UI
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
# 4) Helper: annualize wages (must match training logic)
# ============================================================
def to_annual(amount: float, unit: str) -> float:
    """Convert wage amount to annual equivalent."""
    unit_multipliers = {
        "Year": 1.0,
        "Month": 12.0,
        "Week": 52.0,
        "Hour": 2080.0  # 40 hours/week * 52 weeks
    }
    return amount * unit_multipliers.get(unit, 1.0)

# ============================================================
# 5) Build model input & predict
# ============================================================
if st.button("Estimate Approval Probability"):
    try:
        # Convert inputs to appropriate types
        pw_wage_val = float(pw_wage)
        offer_from_val = float(offer_from)
        offer_to_val = float(offer_to)

        # Calculate derived features
        pw_annual = to_annual(pw_wage_val, pw_unit)
        offer_mid = (offer_from_val + offer_to_val) / 2.0
        offer_annual = to_annual(offer_mid, offer_unit)
        wage_ratio = offer_annual / pw_annual if pw_annual > 0 else 0.0

        # Create input DataFrame with ALL features as strings for categorical columns
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
        
        # Explicitly convert object columns to string dtype
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)

        # Make prediction
        prob_array = model.predict_proba(df)
        prob = float(prob_array[:, 1][0] * 100.0)

        # Display results
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
        
        # Show detailed traceback
        with st.expander("Debug Information (Click to expand)"):
            import traceback
            st.code(traceback.format_exc())
