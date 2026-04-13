import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

# === CUSTOM CSS (Modern, Clean, Like Your 2nd Pic) ===
st.markdown("""
<style>
    .main {background: #f8f9fc; padding: 0;}
    .header {font-weight: 700; font-size: 28px; color: #1e293b; margin: 20px 0 10px;}
    .card {background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 16px;}
    .pill {padding: 6px 16px; border-radius: 999px; font-weight: 600; font-size: 14px; display: inline-block;}
    .pill-allowed {background: #dcfce7; color: #166534; border: 1px solid #bbf7d0;}
    .pill-review {background: #fef3c7; color: #92400e; border: 1px solid #fde68a;}
    .pill-blocked {background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5;}
    .kpi-label {font-weight: 600; color: #475569; font-size: 14px; margin: 8px 0 4px;}
    .kpi-value {font-weight: 700; font-size: 20px; color: #1e293b;}
    .top-match-label {font-weight: 600; color: #64748b; font-size: 14px;}
    .top-match-value {font-size: 15px; color: #1e293b; margin-bottom: 8px;}
    .reason {color: #dc2626; font-weight: 500; font-size: 14px;}
    .xgboost-risk {font-weight: 700; color: #7c3aed; font-size: 18px;}
    .history-table {font-size: 13px;}
    .history-table td, .history-table th {padding: 8px 12px; text-align: left;}
    .history-table th {background: #f1f5f9; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# === SESSION STATE ===
if 'history' not in st.session_state:
    st.session_state.history = []

# === SIDEBAR ===
with st.sidebar:
    st.markdown("### Settings")
    data_path = st.text_input("Data path", "cleaned_aml_data.xlsx")
    match_threshold = st.slider("Match threshold", 0.0, 1.0, 0.76, 0.01)
    show_trans = st.checkbox("Show transliteration & canonical forms", False)
    st.markdown("*Risk probability uses your XGBoost model.*")

# === MAIN UI ===
st.markdown('<div class="header">AML Name Screening</div>', unsafe_allow_html=True)
st.markdown("*Enter a full name (English / Français / العربية). The system returns a decision and model-backed risk probabilities.*")

input_name = st.text_input("Full name to check", placeholder="e.g. Yosra Dabagh")

if st.button("Check", type="primary"):
    start_time = datetime.now()

    # === API CALL WITH RETRY ===
    max_retries = 5
    response = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:5000/predict',
                json={"name": input_name},
                timeout=10
            )
            if response.status_code == 200:
                break
        except:
            if attempt < max_retries - 1:
                st.warning(f"Retrying... ({attempt + 1}/5)")
                time.sleep(2)
            else:
                st.error("API connection failed. Is the Flask server running?")
                st.stop()

    result = response.json()

    # === EXTRACT DATA ===
    decision = result["decision"]
    similarity = float(result["similarity"].strip('%')) if result["similarity"] != "-" else 0
    confidence = float(result["confidence"].strip('%'))
    reason = result["reason"]
    top_match = result["top_match"]
    nationality = result["nationality"]
    risk_category = result["risk_category"]
    notes = result["notes"]
    xgboost_risk = float(result["xgboost_risk"].strip('%')) if result["xgboost_risk"] != "-" else 0

    duration = int((datetime.now() - start_time).total_seconds() * 1000)

    # === RESULT CARD ===
    st.markdown(f'<div class="card"><small>Completed in {duration} ms</small></div>', unsafe_allow_html=True)

    # Decision Pill
    pill_class = {
        "ALLOWED": "pill-allowed",
        "REVIEW": "pill-review",
        "BLOCKED": "pill-blocked"
    }[decision]
    st.markdown(f'<span class="pill {pill_class}">Decision: {decision}</span>', unsafe_allow_html=True)

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="kpi-label">Similarity</div><div class="kpi-value">{similarity:.1f}%</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-label">Confidence</div><div class="kpi-value">{confidence:.1f}%</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-label">Reason</div><div class="reason">{reason}</div>', unsafe_allow_html=True)

    # Top Match Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Top match")
    st.markdown(f'<div class="top-match-label">Name:</div><div class="top-match-value">{top_match}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="top-match-label">Nationality:</div><div class="top-match-value">{nationality}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="top-match-label">Risk Category:</div><div class="top-match-value">{risk_category}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="top-match-label">Notes:</div><div class="top-match-value">{notes}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="top-match-label">XGBoost risk probability:</div><div class="xgboost-risk">{xgboost_risk:.2f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # === HISTORY ===
    st.session_state.history.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Input": input_name,
        "Decision": decision,
        "Similarity %": f"{similarity:.1f}",
        "Risk Category": risk_category,
        "XGBoost Risk %": f"{xgboost_risk:.2f}",
        "Top Match": top_match,
        "Reason": reason
    })

# === HISTORY TABLE ===
st.markdown("### History")
filter_text = st.text_input("Filter...", "")
filtered = [h for h in st.session_state.history if filter_text.lower() in " ".join(h.values()).lower()]
if filtered:
    df = pd.DataFrame(filtered)
    st.markdown(df.to_html(classes="history-table", index=False, escape=False), unsafe_allow_html=True)
else:
    st.info("No history yet.")