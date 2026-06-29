import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="Clinical AI Governance Console", layout="wide")

st.title("🏥 Clinical AI Trust & Governance Interface")
st.write("An engineering solution to machine learning black-boxes and algorithmic bias in healthcare pipelines.")
st.write("---")

@st.cache_resource
def load_production_assets():
    """Loads trained models and data contexts safely."""
    if os.path.exists("models/readmission_model.pkl"):
        with open("models/readmission_model.pkl", "rb") as f:
            model = pickle.load(f)
        X_train = pd.read_csv("data/X_train_sample.csv")
    else:
        st.error("Model assets not detected. Please execute 'python pipeline.py' in your terminal first.")
        st.stop()
        
    explainer = shap.TreeExplainer(model)
    return model, X_train, explainer

model, X_train, explainer = load_production_assets()

# Sidebar: Live Clinical Matrix Configuration
st.sidebar.header("🔬 Live Patient Profile Simulator")
st.sidebar.write("Modify the patient parameters below to see how the mathematical weights adjust instantly.")

age = st.sidebar.slider("Patient Age", 40, 100, 72)
comorb = st.sidebar.slider("Charlson Comorbidity Index (Severity of Illness)", 0, 10, 4)
meds = st.sidebar.slider("Total Active Medications Prescribed", 1, 30, 14)
prior = st.sidebar.slider("Prior Institutional Hospitalizations (Past Year)", 0, 5, 2)
dual = st.sidebar.selectbox("Dual-Eligible Status (Socio-Economic Indicator)", [0, 1], 
                            format_func=lambda x: "Yes (Low Income / Subsidized)" if x==1 else "No")

# Gather parameters into dataframe vector matching layout dimensions
simulated_patient = pd.DataFrame([[age, comorb, meds, prior, dual]], columns=X_train.columns)

# Calculate live risk evaluation 
calculated_risk_prob = model.predict_proba(simulated_patient)[0][1]

# Display Segment Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔮 Model Risk Allocation")
    st.metric(label="Calculated Readmission Probability", value=f"{calculated_risk_prob*100:.1f}%")
    
    if calculated_risk_prob > 0.60:
        st.error("⚠️ CRITICAL ALERT: Model suggests intense post-discharge clinical coordination required.")
    elif calculated_risk_prob > 0.35:
        st.warning("⚡ WARNING: Moderate baseline risk. Flagged for intermediate care transition tracking.")
    else:
        st.success("✅ CLEAR: Standard care transition track authorized.")

with col2:
    st.header("⚖️ Systemic Audit Metrics")
    try:
        with open("docs/audit_results.json", "r") as f:
            audit = json.load(f)
        
        st.metric(label="Disparate Impact Ratio (4/5ths Rule Verification)", 
                  value=audit['disparate_impact_ratio'], 
                  delta=f"Disposition: {audit['status'] if 'status' in audit else audit['audit_disposition']}")
        st.caption("Values below 0.80 signal a structural bias violation, triggering code intervention requirements.")
    except Exception:
        st.info("Audit data logging records missing. Re-run backend pipeline to restore telemetry.")

st.write("---")
st.header("🔍 Post-Hoc Game-Theoretic Attribution (SHAP Explainer Matrix)")
st.write("Instead of rendering an opaque, raw decision score, this visualization maps exactly how much each individual feature shifted the model's ultimate prediction away from baseline expectations.")

# Calculate local shap feature distributions for individual case profile
individual_shap_values = explainer(simulated_patient)

# Generate and capture explicit feature weights plot
fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.bar(individual_shap_values[0], show=False)
plt.tight_layout()
st.pyplot(fig)
