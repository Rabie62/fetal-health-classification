import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="CardioCare: Fetal Health Classifier",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(SCRIPT_DIR, "fetal_health_bundle.joblib")
DATA_PATH = os.path.join(SCRIPT_DIR, "fetal_health.csv")

@st.cache_resource
def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        st.error("Model bundle not found!")
        st.stop()
    return joblib.load(BUNDLE_PATH)

bundle = load_bundle()
models = bundle['models']
eval_results = bundle['eval_results']
best_params = bundle.get('best_params', {})
feature_importance = pd.DataFrame(bundle['feature_importance'])
feature_summary = bundle['feature_summary']

CLASS_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathological"}

# App Title
st.title("👶 CardioCare: Fetal Health CTG Classifier")

# Sidebar
with st.sidebar:
    st.header("Tuned Model Selection")
    
    selected_model_name = st.selectbox(
        "Select Model:",
        list(models.keys()),
        index=0
    )
    current_model = models[selected_model_name]
    m_info = eval_results[selected_model_name]

    st.markdown("---")
    st.subheader("Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("5-Fold CV F1", f"{m_info['5-Fold CV F1-Macro']:.3f}")
    c2.metric("Accuracy", f"{m_info['Accuracy']:.1%}")
    c1.metric("Macro F1", f"{m_info['Macro F1']:.3f}")
    c2.metric("Pathological Recall", f"{m_info['Pathological Sensitivity']:.1%}")

tabs = st.tabs([
    "🩺 Bedside CTG Evaluation",
    "⚙️ Model Benchmarks and Optimal Parameters",
    "📈 Feature Importance Rankings",
    "📂 Explore Clinical Cohort Data"
])

# ---------------- TAB 1: Bedside CTG Evaluation ----------------
with tabs[0]:
    st.subheader(f"Cardiotocogram Assessment (`{selected_model_name}`)")
    
    preset = st.selectbox(
        "Quick Load Clinical Case Preset:",
        [
            "Custom CTG Entry",
            "Case 1: Normal Fetal Status (High Accelerations, Normal Baseline)",
            "Case 2: Suspect Fetal Status (Reduced Variability, Moderate Contractions)",
            "Case 3: Pathological Fetal Distress (Prolonged Decelerations, Severe Abnormal Variability)"
        ]
    )

    defaults = {
        'baseline value': 133.0,
        'accelerations': 0.003,
        'fetal_movement': 0.0,
        'uterine_contractions': 0.006,
        'light_decelerations': 0.0,
        'severe_decelerations': 0.0,
        'prolongued_decelerations': 0.0,
        'abnormal_short_term_variability': 35.0,
        'mean_value_of_short_term_variability': 1.2,
        'percentage_of_time_with_abnormal_long_term_variability': 0.0,
        'mean_value_of_long_term_variability': 8.5,
        'histogram_width': 64.0,
        'histogram_min': 105.0,
        'histogram_max': 169.0,
        'histogram_number_of_peaks': 3.0,
        'histogram_number_of_zeroes': 0.0,
        'histogram_mode': 135.0,
        'histogram_mean': 137.0,
        'histogram_median': 136.0,
        'histogram_variance': 10.0,
        'histogram_tendency': 0.0
    }

    if preset == "Case 1: Normal Fetal Status (High Accelerations, Normal Baseline)":
        defaults.update({
            'baseline value': 132.0, 'accelerations': 0.006, 'fetal_movement': 0.0,
            'uterine_contractions': 0.006, 'light_decelerations': 0.0, 'severe_decelerations': 0.0,
            'prolongued_decelerations': 0.0, 'abnormal_short_term_variability': 20.0,
            'mean_value_of_short_term_variability': 2.0, 'percentage_of_time_with_abnormal_long_term_variability': 0.0,
            'mean_value_of_long_term_variability': 10.4, 'histogram_variance': 8.0
        })
    elif preset == "Case 2: Suspect Fetal Status (Reduced Variability, Moderate Contractions)":
        defaults.update({
            'baseline value': 140.0, 'accelerations': 0.0, 'fetal_movement': 0.001,
            'uterine_contractions': 0.004, 'light_decelerations': 0.001, 'severe_decelerations': 0.0,
            'prolongued_decelerations': 0.0, 'abnormal_short_term_variability': 65.0,
            'mean_value_of_short_term_variability': 0.6, 'percentage_of_time_with_abnormal_long_term_variability': 22.0,
            'mean_value_of_long_term_variability': 4.5, 'histogram_variance': 16.0
        })
    elif preset == "Case 3: Pathological Fetal Distress (Prolonged Decelerations, Severe Abnormal Variability)":
        defaults.update({
            'baseline value': 130.0, 'accelerations': 0.0, 'fetal_movement': 0.002,
            'uterine_contractions': 0.002, 'light_decelerations': 0.003, 'severe_decelerations': 0.001,
            'prolongued_decelerations': 0.004, 'abnormal_short_term_variability': 84.0,
            'mean_value_of_short_term_variability': 0.4, 'percentage_of_time_with_abnormal_long_term_variability': 60.0,
            'mean_value_of_long_term_variability': 2.1, 'histogram_variance': 55.0
        })

    st.markdown("#### Cardiotocogram Vital Signs and Variability")
    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
    with r1_c1:
        fhr = st.number_input("Fetal Heart Rate Baseline (bpm)", 100.0, 180.0, float(defaults['baseline value']), step=1.0)
    with r1_c2:
        acc = st.number_input("Accelerations (/sec)", 0.0, 0.03, float(defaults['accelerations']), step=0.001, format="%.4f")
    with r1_c3:
        uc = st.number_input("Uterine Contractions (/sec)", 0.0, 0.03, float(defaults['uterine_contractions']), step=0.001, format="%.4f")
    with r1_c4:
        fm = st.number_input("Fetal Movement (/sec)", 0.0, 0.5, float(defaults['fetal_movement']), step=0.001, format="%.4f")

    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
    with r2_c1:
        astv = st.number_input("Abnormal Short-Term Var (%)", 0.0, 100.0, float(defaults['abnormal_short_term_variability']), step=1.0)
    with r2_c2:
        mstv = st.number_input("Mean Short-Term Var", 0.0, 10.0, float(defaults['mean_value_of_short_term_variability']), step=0.1)
    with r2_c3:
        altv = st.number_input("Abnormal Long-Term Var (%)", 0.0, 100.0, float(defaults['percentage_of_time_with_abnormal_long_term_variability']), step=1.0)
    with r2_c4:
        mltv = st.number_input("Mean Long-Term Var", 0.0, 60.0, float(defaults['mean_value_of_long_term_variability']), step=0.5)

    with st.expander("Decelerations and Histogram Features (Advanced)", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            dl = st.number_input("Light Decelerations (/sec)", 0.0, 0.03, float(defaults['light_decelerations']), step=0.001, format="%.4f")
            h_width = st.number_input("Histogram Width", 0.0, 250.0, float(defaults['histogram_width']))
        with d2:
            severe_dec = st.number_input("Severe Decelerations (/sec)", 0.0, 0.01, float(defaults['severe_decelerations']), step=0.001, format="%.4f")
            h_min = st.number_input("Histogram Min", 0.0, 200.0, float(defaults['histogram_min']))
        with d3:
            prolong_dec = st.number_input("Prolonged Decelerations (/sec)", 0.0, 0.01, float(defaults['prolongued_decelerations']), step=0.001, format="%.4f")
            h_max = st.number_input("Histogram Max", 50.0, 250.0, float(defaults['histogram_max']))
        with d4:
            h_variance = st.number_input("Histogram Variance", 0.0, 300.0, float(defaults['histogram_variance']))
            h_tendency = st.selectbox("Histogram Tendency", [-1.0, 0.0, 1.0], index=1)

    patient_record = {
        'baseline value': fhr,
        'accelerations': acc,
        'fetal_movement': fm,
        'uterine_contractions': uc,
        'light_decelerations': dl,
        'severe_decelerations': severe_dec,
        'prolongued_decelerations': prolong_dec,
        'abnormal_short_term_variability': astv,
        'mean_value_of_short_term_variability': mstv,
        'percentage_of_time_with_abnormal_long_term_variability': altv,
        'mean_value_of_long_term_variability': mltv,
        'histogram_width': h_width,
        'histogram_min': h_min,
        'histogram_max': h_max,
        'histogram_number_of_peaks': defaults['histogram_number_of_peaks'],
        'histogram_number_of_zeroes': defaults['histogram_number_of_zeroes'],
        'histogram_mode': defaults['histogram_mode'],
        'histogram_mean': defaults['histogram_mean'],
        'histogram_median': defaults['histogram_median'],
        'histogram_variance': h_variance,
        'histogram_tendency': h_tendency
    }

    patient_df = pd.DataFrame([patient_record])[bundle['features']]

    # Model inference
    if selected_model_name in ['XGBoost', 'LightGBM']:
        pred_label_idx = int(current_model.predict(patient_df)[0]) + 1
        probs = current_model.predict_proba(patient_df)[0]
    else:
        pred_label_idx = int(current_model.predict(patient_df)[0])
        probs = current_model.predict_proba(patient_df)[0]

    st.markdown("---")
    res1, spacer, res2 = st.columns([1.2, 0.4, 2.0])

    with res1:
        st.subheader("Prediction Result")
        if pred_label_idx == 3:
            st.error("🚨 **PATHOLOGICAL FETAL DISTRESS (Class 3)**")
            st.markdown("**Action:** Immediate obstetric evaluation, emergency scalp lactate test, and prepare for urgent delivery.")
        elif pred_label_idx == 2:
            st.warning("⚠️ **SUSPECT FETAL STATUS (Class 2)**")
            st.markdown("**Action:** Reposition maternal patient, maternal oxygenation, continuous CTG monitoring, and re-assess in 15 mins.")
        else:
            st.success("✅ **NORMAL FETAL HEALTH (Class 1)**")
            st.markdown("**Action:** Reassuring tracing. Continue routine intrapartum monitoring.")

        prob_df = pd.DataFrame({
            'Status': ['Normal', 'Suspect', 'Pathological'],
            'Probability': [f"{p:.1%}" for p in probs]
        })
        st.table(prob_df)

    with res2:
        st.subheader("Key Clinical Indicators")
        b1, b2, b3 = st.columns(3)
        b1.metric("Short-Term Var (ASTV)", f"{astv:.0f}%", delta="Critical (>60%)" if astv > 60 else "Normal", delta_color="inverse")
        b2.metric("Prolonged Decelerations", f"{prolong_dec:.4f}", delta="Severe Distress" if prolong_dec > 0 else "None", delta_color="inverse")
        b3.metric("FHR Baseline", f"{fhr:.0f} bpm", delta="Tachycardia (>160)" if fhr > 160 else ("Bradycardia (<110)" if fhr < 110 else "Reassuring"), delta_color="normal" if 110 <= fhr <= 160 else "inverse")

# ---------------- TAB 2: Model Benchmarks & Optimal Parameters ----------------
with tabs[1]:
    st.subheader("5-Fold Stratified GridSearchCV Performance")
    
    col_bench1, col_bench2 = st.columns([1.6, 1.4])
    with col_bench1:
        st.markdown("#### Cross-Validation and Test Holdout Scores")
        bench_df = pd.DataFrame(eval_results).T.reset_index().rename(columns={'index': 'Model'})
        st.dataframe(bench_df, use_container_width=True)

    with col_bench2:
        st.markdown("#### Optimal Parameters (via GridSearch)")
        for m_name, params in best_params.items():
            with st.expander(f"🔹 {m_name}", expanded=True):
                st.write(params)

# ---------------- TAB 3: Feature Importance ----------------
with tabs[2]:
    st.subheader("Clinical Feature Importance Rankings")
    fig_full_fi = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale="Purples"
    )
    fig_full_fi.update_layout(yaxis={'categoryorder': 'total ascending'}, height=480)
    st.plotly_chart(fig_full_fi, use_container_width=True)

# ---------------- TAB 4: Explore Data ----------------
with tabs[3]:
    st.subheader("Explore the Fetal Health Clinical Cohort")
    if os.path.exists(DATA_PATH):
        raw_df = pd.read_csv(DATA_PATH)
        st.dataframe(raw_df, use_container_width=True)
