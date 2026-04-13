# api.py - AML Name Screening Backend with Custom Arabic Transliteration

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from jellyfish import jaro_winkler_similarity, levenshtein_distance, soundex
from joblib import load
import re
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- Custom Arabic to Latin Transliteration Map ---
arabic_map = {
    'ا':'a','أ':'a','إ':'i','آ':'a','ب':'b','ت':'t','ث':'th','ج':'j','ح':'h','خ':'kh',
    'د':'d','ذ':'dh','ر':'r','ز':'z','س':'s','ش':'sh','ص':'s','ض':'d','ط':'t','ظ':'z',
    'ع':'a','غ':'gh','ف':'f','ق':'q','ك':'k','ل':'l','م':'m','ن':'n','ه':'h','ة':'a',
    'و':'w','ؤ':'u','ي':'y','ئ':'i','ى':'a','‎ٰ':'','ﻻ':'la','لا':'la','ٔ':'','ٕ':'',
    '٠':'0','١':'1','٢':'2','٣':'3','٤':'4','٥':'5','٦':'6','٧':'7','٨':'8','٩':'9'
}

# --- LOAD & PREPROCESS WATCHLIST (Transliterate Arabic Names) ---
WATCHLIST_PATH = r"..\dataset\cleaned_aml_data.xlsx"
MODEL_PATH = r"..\models\aml_model_xgboost.joblib"
SCALER_PATH = r"..\models\aml_model_scaler.joblib"

if not os.path.exists(WATCHLIST_PATH):
    raise FileNotFoundError(f"Watchlist not found: {WATCHLIST_PATH}")

watchlist_df = pd.read_excel(WATCHLIST_PATH)
print(f"Watchlist loaded: {watchlist_df.shape[0]} entries")

# Transliterate Arabic names to Latin
arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
translated_df = watchlist_df.copy()  # Make a copy for transliterated versions
for i, name in enumerate(translated_df['Full Name']):
    name_str = str(name).strip()
    if arabic_pattern.search(name_str):
        try:
            transliterated = ''.join(arabic_map.get(c, c) for c in name_str).strip()
            if transliterated and transliterated != name_str:
                print(f"Transliterated '{name_str}' → '{transliterated}'")
                translated_df.at[i, 'Full Name'] = transliterated
        except Exception as e:
            print(f"Transliteration error for '{name_str}': {e}")

# Now we have two DFs: original (for display) + translated (for matching)

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Model or scaler missing: {MODEL_PATH}, {SCALER_PATH}")

model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
print("XGBoost model and scaler loaded.")

# --- Helper: Normalize Name ---
def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    name = re.sub(r'\s+', ' ', name)  # Normalize spaces
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = name.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('á', 'a')  # Diacritics
    return name

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_name = data.get('name', '').strip()
    if not input_name:
        return jsonify({"error": "Empty name"}), 400

    print(f"\nInput: {input_name}")

    name1 = normalize_name(input_name)

    best_combined = 0.0
    best_match = None
    best_features = None
    best_jw = 0.0
    best_lev_score = 0.0
    best_soundex = 0

    # --- Iterate over translated watchlist for matching, but use original for display ---
    for idx, row in translated_df.iterrows():
        name2 = normalize_name(row["Full Name"])

        # Jaro-Winkler
        jw = jaro_winkler_similarity(name1, name2)

        # Levenshtein (normalized)
        lev_dist = levenshtein_distance(name1, name2)
        max_len = max(len(name1), len(name2))
        lev_score = (max_len - lev_dist) / max_len if max_len > 0 else 0

        # Soundex
        sx1 = soundex(name1)
        sx2 = soundex(name2)
        soundex_match = 1 if sx1 == sx2 and sx1 != '0000' else 0

        # Combined
        combined = (jw * 0.5) + (lev_score * 0.3) + (soundex_match * 0.2)

        if combined > best_combined:
            best_combined = combined
            best_jw = jw
            best_lev_score = lev_score
            best_soundex = soundex_match
            best_match = watchlist_df.iloc[idx]  # Use ORIGINAL row for display

    # --- Default: No match ---
    if best_match is None or best_combined < 0.3:
        return jsonify({
            "decision": "ALLOWED",
            "confidence": "100.0%",
            "similarity": "-",
            "xgboost_risk": "-",
            "top_match": "-",
            "nationality": "-",
            "risk_category": "-",
            "notes": "-",
            "reason": "No significant risk identified."
        })

    # --- Extract best match info (from ORIGINAL DF) ---
    top_match_name = str(best_match['Full Name'])
    nationality = str(best_match.get('Nationality', '-'))
    risk_category = str(best_match.get('Risk Category', '-'))
    notes = str(best_match.get('Notes', '-'))

    # --- Risk flags ---
    rc_lower = risk_category.lower()
    is_high_risk = 1 if any(word in rc_lower for word in ["terrorism", "laundering", "sanction"]) else 0
    is_medium_risk = 1 if "pep" in rc_lower else 0
    has_nationality_match = 1 if nationality.lower() in name1 else 0

    # --- Feature vector for XGBoost ---
    feature_vector = np.array([[
        best_jw,
        best_lev_score,
        best_soundex,
        is_high_risk,
        is_medium_risk,
        has_nationality_match
    ]])
    feature_vector_scaled = scaler.transform(feature_vector)
    pred = model.predict(feature_vector_scaled)[0]
    pred_proba = model.predict_proba(feature_vector_scaled)[0]
    reverse_mapping = {0: "ALLOWED", 1: "REVIEW", 2: "BLOCKED"}
    decision = reverse_mapping[pred]
    confidence = pred_proba[pred] * 100
    block_prob = pred_proba[2] * 100

    # --- Reason ---
    if decision == "REVIEW":
        reason = "Moderate similarity to known entity."
    elif decision == "BLOCKED":
        reason = "High risk match detected."
    else:
        reason = "No significant risk identified."

    # --- Final Response ---
    result = {
        "decision": decision,
        "confidence": f"{confidence:.1f}%",
        "similarity": f"{best_combined * 100:.2f}%" if decision != "ALLOWED" else "-",
        "risk_category": risk_category if decision != "ALLOWED" else "-",
        "xgboost_risk": f"{block_prob:.2f}%" if decision != "ALLOWED" else "-",
        "top_match": top_match_name if decision != "ALLOWED" else "-",
        "nationality": nationality if decision != "ALLOWED" else "-",
        "notes": notes if decision != "ALLOWED" else "-",
        "reason": reason
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)