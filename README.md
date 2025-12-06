# PERM EB2 Approval Predictor

A machine learning application that predicts the approval probability for PERM (Program Electronic Review Management) EB2 labor certification applications.

## 📋 Project Overview

This project uses historical PERM data (2022-2024) to train classification models that predict whether a PERM EB2 application will be approved or denied. The best model is selected using **F1 Score** as the primary metric.

### Why F1 Score?

Per professor's recommendation, F1 Score was chosen over ROC-AUC because:
- F1 is the harmonic mean of precision and recall
- Better suited for predicting approval probability (classification task)
- Balances false positives and false negatives
- More appropriate when predicting binary outcomes

## 🗂️ Project Structure

```
perm-eb2-predictor/
│
├── Perm_EB2_F1_Optimized.ipynb   # Jupyter notebook (training code)
├── app.py                         # Flask web application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── best_model_for_deployment.pkl  # Trained model (generated from notebook)
├── model_results_optimized.csv    # All model results (generated)
└── summary_report.txt             # Model summary (generated)
```

## 🚀 Quick Start

### 1. Train the Model (Google Colab)

1. Open `Perm_EB2_F1_Optimized.ipynb` in Google Colab
2. Upload your PERM data files (2022, 2023, 2024 Excel files)
3. Run all cells
4. Download `best_model_for_deployment.pkl`

### 2. Run the Web App Locally

```bash
# Clone or download the project
cd perm-eb2-predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your trained model in the same directory
# (best_model_for_deployment.pkl)

# Run the app
python app.py
```

Open http://localhost:5000 in your browser.

### 3. Deploy to Cloud (Render, Heroku, Railway)

#### Render (Recommended - Free Tier)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Add your `best_model_for_deployment.pkl` to the repo
6. Deploy!

#### Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
git push heroku main
```

## 📊 Model Features

The model uses the following features:

| Feature | Description |
|---------|-------------|
| `PW_WAGE_ANNUAL` | Prevailing wage (annualized) |
| `OFFER_WAGE_ANNUAL` | Offered wage (annualized) |
| `WAGE_RATIO` | Offered wage / Prevailing wage |
| `WAGE_PREMIUM` | Offered wage - Prevailing wage |
| `LOG_PW_WAGE` | Log-transformed prevailing wage |
| `LOG_OFFER_WAGE` | Log-transformed offered wage |
| `EDUCATION_LEVEL` | Minimum education (0-5 scale) |
| `HAS_OWNERSHIP` | Foreign worker ownership interest |
| `DATA_YEAR` | Filing year |
| `SOC_MAJOR` | Occupation category (first 2 digits) |
| `NAICS_SECTOR` | Industry sector (first 2 digits) |
| `REGION` | US region (Northeast/Midwest/South/West) |
| `WORKSITE_STATE` | State where job is located |

## 🤖 Models Trained

The notebook trains and evaluates the following models:

- ✅ Decision Tree
- ✅ Logistic Regression
- ✅ Support Vector Machine (SVM)
- ✅ Neural Network (MLP)
- ✅ Naive Bayes
- ✅ Random Forest
- ✅ XGBoost
- ✅ LightGBM
- ✅ K-Nearest Neighbors (KNN)
- ✅ Gradient Boosting
- ✅ AdaBoost

Each model is tested with multiple hyperparameter configurations (40+ total variations).

## 🔌 API Usage

The app provides a REST API endpoint:

```bash
# POST /api/predict
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pw_wage": 100000,
    "offer_wage_from": 120000,
    "offer_wage_to": 150000,
    "education": "masters",
    "state": "CA",
    "soc_code": "15-1252",
    "naics_code": "541511",
    "ownership": "N",
    "year": 2024
  }'

# Response
{
  "approval_probability": 85.3,
  "prediction": "APPROVED"
}
```

### Health Check

```bash
curl http://localhost:5000/health

# Response
{"status": "healthy", "model_loaded": true}
```

## 📈 Target Definition

- **Approved (1):** Certified + Certified-Expired
  - Both indicate USCIS approved the labor certification
  - "Certified-Expired" means employer didn't file I-140 in time
- **Denied (0):** Denied + Withdrawn

## 🏢 Use Case: Law Firm Case Prioritization

| Probability | Risk Level | Action |
|-------------|------------|--------|
| ≥ 80% | LOW RISK | Standard processing |
| 60-79% | MODERATE RISK | Review case details |
| < 60% | HIGH RISK | Detailed review before filing |

## 📦 Dependencies

```
flask==3.0.0
gunicorn==21.2.0
scikit-learn==1.5.1
xgboost==2.0.3
lightgbm==4.3.0
pandas==2.2.0
numpy==1.26.4
joblib==1.3.2
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5000 | Server port |
| `FLASK_DEBUG` | False | Debug mode |

### Databricks MLflow (Optional)

The notebook includes Databricks MLflow integration for experiment tracking:

```python
# Set in Google Colab secrets
DATABRICKS_HOST = "your-databricks-host"
DATABRICKS_TOKEN = "your-databricks-token"
```

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Note:** This prediction tool is based on historical data and should be used as one factor in case assessment, not as the sole decision-making criterion.
