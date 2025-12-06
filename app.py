"""
PERM EB2 Approval Prediction - Flask Web Application
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# =================================================================================
# LOAD MODEL
# =================================================================================
MODEL_PATH = 'best_model_for_deployment.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found at {MODEL_PATH}")
    model = None


# =================================================================================
# CONFIGURATION
# =================================================================================
EDUCATION_MAP = {
    "none": 0, "high_school": 1, "associates": 2,
    "bachelors": 3, "masters": 4, "doctorate": 5, "other": 2
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

US_STATES = [
    ('AL', 'Alabama'), ('AK', 'Alaska'), ('AZ', 'Arizona'), ('AR', 'Arkansas'),
    ('CA', 'California'), ('CO', 'Colorado'), ('CT', 'Connecticut'), ('DE', 'Delaware'),
    ('DC', 'District of Columbia'), ('FL', 'Florida'), ('GA', 'Georgia'), ('HI', 'Hawaii'),
    ('ID', 'Idaho'), ('IL', 'Illinois'), ('IN', 'Indiana'), ('IA', 'Iowa'),
    ('KS', 'Kansas'), ('KY', 'Kentucky'), ('LA', 'Louisiana'), ('ME', 'Maine'),
    ('MD', 'Maryland'), ('MA', 'Massachusetts'), ('MI', 'Michigan'), ('MN', 'Minnesota'),
    ('MS', 'Mississippi'), ('MO', 'Missouri'), ('MT', 'Montana'), ('NE', 'Nebraska'),
    ('NV', 'Nevada'), ('NH', 'New Hampshire'), ('NJ', 'New Jersey'), ('NM', 'New Mexico'),
    ('NY', 'New York'), ('NC', 'North Carolina'), ('ND', 'North Dakota'), ('OH', 'Ohio'),
    ('OK', 'Oklahoma'), ('OR', 'Oregon'), ('PA', 'Pennsylvania'), ('RI', 'Rhode Island'),
    ('SC', 'South Carolina'), ('SD', 'South Dakota'), ('TN', 'Tennessee'), ('TX', 'Texas'),
    ('UT', 'Utah'), ('VT', 'Vermont'), ('VA', 'Virginia'), ('WA', 'Washington'),
    ('WV', 'West Virginia'), ('WI', 'Wisconsin'), ('WY', 'Wyoming')
]


# =================================================================================
# HELPER FUNCTION
# =================================================================================
def preprocess_input(form_data):
    pw_wage = float(form_data.get('pw_wage', 0))
    offer_wage_from = float(form_data.get('offer_wage_from', 0))
    offer_wage_to = float(form_data.get('offer_wage_to', 0) or offer_wage_from)
    
    offer_wage = (offer_wage_from + offer_wage_to) / 2
    wage_ratio = offer_wage / pw_wage if pw_wage > 0 else 1
    wage_premium = offer_wage - pw_wage
    
    education = form_data.get('education', 'bachelors').lower()
    state = form_data.get('state', 'CA').upper()
    
    return pd.DataFrame([{
        'PW_WAGE_ANNUAL': pw_wage,
        'OFFER_WAGE_ANNUAL': offer_wage,
        'WAGE_RATIO': wage_ratio,
        'WAGE_PREMIUM': wage_premium,
        'LOG_PW_WAGE': np.log1p(pw_wage),
        'LOG_OFFER_WAGE': np.log1p(offer_wage),
        'EDUCATION_LEVEL': EDUCATION_MAP.get(education, 2),
        'HAS_OWNERSHIP': 1 if form_data.get('ownership', 'N') == 'Y' else 0,
        'DATA_YEAR': int(form_data.get('year', 2024)),
        'SOC_MAJOR': str(form_data.get('soc_code', '15'))[:2],
        'NAICS_SECTOR': str(form_data.get('naics_code', '54'))[:2],
        'REGION': STATE_REGIONS.get(state, 'West'),
        'WORKSITE_STATE': state
    }])


# =================================================================================
# HTML TEMPLATES
# =================================================================================
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PERM EB2 Approval Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 700px; margin: 0 auto; background: white; border-radius: 16px;
                     box-shadow: 0 20px 60px rgba(0,0,0,0.3); padding: 40px; }
        h1 { text-align: center; color: #1e3c72; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }
        .badge { display: inline-block; background: #2a5298; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #333; font-weight: 500; }
        input, select { width: 100%; padding: 12px; border: 2px solid #e1e1e1; border-radius: 8px; font-size: 16px; }
        input:focus, select:focus { outline: none; border-color: #2a5298; }
        .row { display: flex; gap: 15px; }
        .row .form-group { flex: 1; }
        button { width: 100%; padding: 15px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                 color: white; border: none; border-radius: 8px; font-size: 18px; font-weight: 600; cursor: pointer; }
        button:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(30, 60, 114, 0.4); }
        .info { background: #f0f4f8; padding: 15px; border-radius: 8px; margin-top: 20px; font-size: 13px; color: #555; }
        .help-text { font-size: 12px; color: #888; margin-top: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏛️ PERM EB2 Approval Predictor</h1>
        <p class="subtitle">Predict approval probability for PERM labor certification<br><span class="badge">Model optimized for F1 Score</span></p>
        
        <form action="/predict" method="POST">
            <div class="form-group">
                <label>Prevailing Wage (Annual $)</label>
                <input type="number" name="pw_wage" placeholder="e.g., 100000" required>
            </div>
            <div class="row">
                <div class="form-group">
                    <label>Offered Wage From ($)</label>
                    <input type="number" name="offer_wage_from" placeholder="e.g., 120000" required>
                </div>
                <div class="form-group">
                    <label>Offered Wage To ($)</label>
                    <input type="number" name="offer_wage_to" placeholder="e.g., 150000">
                </div>
            </div>
            <div class="form-group">
                <label>Minimum Education Required</label>
                <select name="education">
                    <option value="bachelors">Bachelor's Degree</option>
                    <option value="masters">Master's Degree</option>
                    <option value="doctorate">Doctorate</option>
                    <option value="associates">Associate's Degree</option>
                    <option value="high_school">High School</option>
                    <option value="none">None</option>
                </select>
            </div>
            <div class="row">
                <div class="form-group">
                    <label>Worksite State</label>
                    <select name="state">
                        {% for code, name in states %}<option value="{{ code }}" {% if code == 'CA' %}selected{% endif %}>{{ name }}</option>{% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Filing Year</label>
                    <select name="year">
                        <option value="2025">2025</option>
                        <option value="2024" selected>2024</option>
                    </select>
                </div>
            </div>
            <div class="row">
                <div class="form-group">
                    <label>SOC Code</label>
                    <input type="text" name="soc_code" placeholder="e.g., 15-1252" value="15-1252">
                </div>
                <div class="form-group">
                    <label>NAICS Code</label>
                    <input type="text" name="naics_code" placeholder="e.g., 541511" value="54">
                </div>
            </div>
            <div class="form-group">
                <label>Foreign Worker Has Ownership Interest?</label>
                <select name="ownership"><option value="N">No</option><option value="Y">Yes</option></select>
            </div>
            <button type="submit">🔮 Predict Approval Probability</button>
        </form>
        <div class="info"><strong>Note:</strong> Based on historical PERM data (2022-2024). Model selected using F1 Score.</div>
    </div>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
               background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 16px;
                     box-shadow: 0 20px 60px rgba(0,0,0,0.3); padding: 40px; text-align: center; }
        h1 { color: #1e3c72; margin-bottom: 30px; }
        .probability-circle { width: 200px; height: 200px; border-radius: 50%; margin: 0 auto 30px;
                              display: flex; align-items: center; justify-content: center;
                              flex-direction: column; font-size: 48px; font-weight: bold; color: white; }
        .green { background: linear-gradient(135deg, #00b894, #00cec9); }
        .orange { background: linear-gradient(135deg, #fdcb6e, #e17055); }
        .red { background: linear-gradient(135deg, #e74c3c, #c0392b); }
        .probability-circle span { font-size: 14px; font-weight: normal; }
        .risk-badge { display: inline-block; padding: 10px 25px; border-radius: 25px; font-weight: 600; margin-bottom: 20px; }
        .risk-badge.green { background: #d4edda; color: #155724; }
        .risk-badge.orange { background: #fff3cd; color: #856404; }
        .risk-badge.red { background: #f8d7da; color: #721c24; }
        .recommendation { background: #f0f4f8; padding: 20px; border-radius: 10px; margin: 20px 0; color: #555; }
        .back-btn { display: inline-block; margin-top: 30px; padding: 15px 40px;
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    color: white; text-decoration: none; border-radius: 8px; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 Prediction Result</h1>
        <div class="probability-circle {{ risk_color }}">{{ probability }}%<span>Approval Probability</span></div>
        <div class="risk-badge {{ risk_color }}">{{ risk_level }}</div>
        <div class="recommendation"><strong>Recommendation:</strong><br>{{ recommendation }}</div>
        <a href="/" class="back-btn">← Make Another Prediction</a>
    </div>
</body>
</html>
"""

ERROR_HTML = """
<!DOCTYPE html>
<html><head><title>Error</title>
<style>body{font-family:sans-serif;background:#1e3c72;min-height:100vh;display:flex;align-items:center;justify-content:center;}
.container{background:white;padding:40px;border-radius:16px;text-align:center;}
h1{color:#e74c3c;}a{display:inline-block;padding:15px 40px;background:#1e3c72;color:white;text-decoration:none;border-radius:8px;margin-top:20px;}</style>
</head><body><div class="container"><h1>⚠️ Error</h1><p>{{ error }}</p><a href="/">← Go Back</a></div></body></html>
"""


# =================================================================================
# ROUTES
# =================================================================================
@app.route('/')
def home():
    return render_template_string(INDEX_HTML, states=US_STATES)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(ERROR_HTML, error='Model not loaded.')
    try:
        form_data = request.form.to_dict()
        input_df = preprocess_input(form_data)
        proba = model.predict_proba(input_df)[0]
        approval_probability = round(proba[1] * 100, 1)
        
        if approval_probability >= 80:
            risk_level, risk_color, recommendation = "LOW RISK", "green", "Strong case. Standard processing recommended."
        elif approval_probability >= 60:
            risk_level, risk_color, recommendation = "MODERATE RISK", "orange", "Review case details. Consider strengthening weak areas."
        else:
            risk_level, risk_color, recommendation = "HIGH RISK", "red", "Significant concerns. Detailed review recommended."
        
        return render_template_string(RESULT_HTML, probability=approval_probability,
                                      risk_level=risk_level, risk_color=risk_color, recommendation=recommendation)
    except Exception as e:
        return render_template_string(ERROR_HTML, error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        input_df = preprocess_input(request.get_json())
        proba = model.predict_proba(input_df)[0]
        return jsonify({'approval_probability': round(proba[1] * 100, 2), 'prediction': 'APPROVED' if proba[1] >= 0.5 else 'DENIED'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🏛️ PERM EB2 Predictor running on http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
