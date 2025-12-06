# 🚀 Deploying to Streamlit Cloud - Instructions

## The Problem
Your `app.py` is a **Flask application**, but Streamlit Cloud expects a **Streamlit application**. 
That's why you're seeing port errors - Flask tries to run on port 5000, which conflicts with Streamlit's infrastructure.

## ✅ Solution: Use the Streamlit Version

I've created `streamlit_app.py` which is a complete Streamlit conversion of your Flask app.

## 📋 Steps to Deploy

### 1. Update Your Repository

Replace or rename files in your GitHub repository:

**Option A (Recommended):**
```bash
# Rename the Flask app
mv app.py flask_app.py

# Use the Streamlit version as your main app
mv streamlit_app.py app.py
```

**Option B (Keep both):**
```bash
# Just add streamlit_app.py to your repo
# Keep app.py for local Flask testing
```

### 2. Update requirements.txt

Replace your current `requirements.txt` with:

```txt
streamlit>=1.28.0
scikit-learn==1.5.1
xgboost==2.0.3
lightgbm==4.3.0
pandas==2.2.0
numpy==1.26.4
joblib==1.3.2
```

**Important:** Remove Flask and Gunicorn (not needed for Streamlit Cloud)

### 3. Ensure Model File is Included

Make sure `best_model_for_deployment.pkl` is in your repository root:

```
your-repo/
├── app.py (or streamlit_app.py)
├── requirements.txt
├── best_model_for_deployment.pkl  ← Must be here!
└── README.md
```

### 4. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. **Main file path:** 
   - If you renamed: `app.py`
   - If you kept separate: `streamlit_app.py`
6. Click "Deploy"

### 5. Wait for Deployment

Streamlit will:
- Install dependencies from requirements.txt
- Load your model
- Start the app (usually takes 2-3 minutes)

## 🎯 What Changed?

### Flask → Streamlit Conversion

| Flask | Streamlit |
|-------|-----------|
| `@app.route()` decorators | Direct function calls |
| HTML templates | Streamlit components |
| `request.form` | `st.form()` |
| `render_template()` | `st.write()`, `st.markdown()` |
| Manual CSS | `st.markdown()` with CSS |
| Port 5000 | Managed by Streamlit |

## ⚙️ Features Preserved

✅ Same ML model and predictions  
✅ Same input fields and validation  
✅ Same risk categorization (Low/Moderate/High)  
✅ Same color-coded results  
✅ Improved user experience with Streamlit widgets  

## 🆕 New Features in Streamlit Version

- ✨ Real-time input validation
- 📊 Expandable input summary
- 🎨 Better mobile responsiveness
- 🔄 Automatic model caching (faster reloads)
- 💾 Built-in session state management

## 🐛 Common Issues & Fixes

### Issue: "Module 'best_model_for_deployment' not found"
**Fix:** Ensure the .pkl file is in your repo root (not in a subfolder)

### Issue: "XGBoost warning about model version"
**Fix:** This is just a warning, not an error. The app will still work.

### Issue: "AttributeError: module has no attribute..."
**Fix:** Check that your model was trained with the same library versions in requirements.txt

### Issue: Still seeing Flask errors
**Fix:** Make sure you're using `streamlit_app.py` as the main file, not `app.py` (Flask version)

## 📱 Accessing Your Deployed App

Once deployed, you'll get a URL like:
```
https://your-app-name.streamlit.app
```

You can:
- Share this URL with users
- Embed it in your website
- Add custom domain (Pro plan)

## 🔒 Security Note

Your model file (`best_model_for_deployment.pkl`) will be public in your GitHub repo. 
If this is a concern:

1. Use Streamlit Secrets for sensitive data
2. Consider a private repository (requires Streamlit Pro)
3. Or keep the model file in a private storage and download it at runtime

## 🎉 That's It!

Your app should now work perfectly on Streamlit Cloud. If you still encounter issues, 
check the Streamlit Cloud logs for specific error messages.

## Need Help?

Common commands for debugging:
```bash
# Test locally before deploying
streamlit run streamlit_app.py

# Check Python version
python --version  # Should be 3.8-3.11

# Verify model can be loaded
python -c "import joblib; print(joblib.load('best_model_for_deployment.pkl'))"
```

---

**Pro Tip:** Streamlit auto-reruns when you push changes to GitHub, so you can iterate quickly!
