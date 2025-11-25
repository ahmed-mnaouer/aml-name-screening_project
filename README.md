# AML Name Screening System  
**Arabic-Aware • Dual Engine • Offline**

| Engine             | Tech                             | Speed | Use Case                     |
|--------------------|----------------------------------|-------|------------------------------|
| ML Engine          | XGBoost + Fuzzy + Transliteration| Fast  | High accuracy, final decision|
| Fast Rule Engine   | Fuzzy score + risk rules         | Instant| Quick pre-screening          |

### Key Features
- Custom Arabic → Latin transliteration (zero dependencies)  
  → `يسري دباغ` = `yosri dabegh` = `yusri dabagh`  
- Handles typos, reversed names, prefixes (Al-, Ben-, Ould-, etc.)  
- Decision: **ALLOWED / REVIEW / BLOCKED** + risk %  
- Beautiful Streamlit UI + Flask API  
- 100% local — no data leaves your machine

### Project Files
```
AML-project/
├── backend/
│   └── screening_system_enhanced.py          
│   └── screening_system.py  
│   └── api_backend.py  
├── frontend/
│   └── app_streamlit.py        
├── dataset/
│   └── cleaned_aml_data.xlsx   # Watchlist (sanctions, PEPs, etc.)
├── models/
│   ├── aml_model_xgboost.joblib
│   └── aml_model_scaler.joblib
│   └── aml_model_random_forest.joblib
│   └── aml_model_logistic_regression.joblib
└── README.md
```
### Run (2 terminals)
#### Terminal 1
```bash
python api_backend.py
```
#### Terminal 2
```bash
streamlit run app_streamlit.py
```

### Install
```bash
pip install streamlit flask pandas numpy jellyfish xgboost joblib scikit-learn openpyxl
```

Ahmed Mnaouer • 2025

Fully offline multilingual AML screening (Arabic / French / English)
