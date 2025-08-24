import streamlit as st
import joblib
import numpy as np
import shap
import plotly.graph_objects as go

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown(
    """
    <style>
        /* General */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Banner styling */
        .banner {
            background: linear-gradient(135deg, #1A5276, #154360);
            padding: 30px;
            border-radius: 18px;
            margin-bottom: 25px;
            text-align: center;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
        }
        .big-title {
            font-size: 42px !important;
            color: #FDFEFE;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 18px;
            color: #D5DBDB;
            font-weight: 400;
        }

        /* Section titles */
        h3 {
            color: #154360 !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 20px;
            font-weight: bold;
        }
    </style>

    <div class="banner">
        <p class="big-title">🩺 Diabetes Risk Prediction Dashboard</p>
        <p class="subtitle">AI-powered prediction with SHAP explanations, risk visualization, and WHO-based health tips</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD MODEL --------------------
model = joblib.load("random_forest_model.joblib")
label_encoder_classes = ['Diabetes', 'Non-Diabetes', 'Pre-Diabetes']

# -------------------- USER INPUT --------------------
st.markdown("### 📋 Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])   
    fbs = st.slider("Fasting Blood Sugar (mg/dL)", 50.0, 300.0, 100.0)
    bmi = st.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0)

with col2:
    wc = st.slider("Waist Circumference (cm)", 50.0, 150.0, 90.0)
    hc = st.slider("Hip Circumference (cm)", 70.0, 150.0, 100.0)
    smoking = st.selectbox("🚬 Tobacco Smoking", ["Yes", "No"])
    alcohol = st.selectbox("🍷 Alcohol Consumption", ["Yes", "No"])
    physical_activity = st.selectbox("🏃 Physical Activity Level", ["Low", "Moderate", "High"])
    family_history = st.selectbox("👨‍👩‍👧 Family History of Diabetes", ["Yes", "No"])

# Model input
features = np.array([[fbs, bmi, age, wc, hc]])
feature_names = ['FBS', 'BMI', 'Age', 'WC', 'HC']

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Now", use_container_width=True):
    prediction_encoded = model.predict(features)
    predicted_status = label_encoder_classes[prediction_encoded[0]]

    prediction_proba = model.predict_proba(features)
    max_prob = np.max(prediction_proba)

    # Risk Status
    if predicted_status == "Diabetes":
        risk_status = "Diabetes (High Risk)" if max_prob >= 0.7 else "Diabetes (Low Risk)"
    elif predicted_status == "Pre-Diabetes":
        risk_status = "Pre-Diabetes (High Risk)" if max_prob >= 0.6 else "Pre-Diabetes (Low Risk)"
    else:
        risk_status = "Non-Diabetes (Low Risk)"

    # -------------------- RESULTS --------------------
    st.markdown("## ✅ Prediction Results")
    st.success(f"**Predicted Status: {risk_status}**")

    st.markdown("### 📊 Prediction Probabilities")
    prob_cols = st.columns(len(label_encoder_classes))
    for i, class_name in enumerate(label_encoder_classes):
        prob_cols[i].metric(class_name, f"{prediction_proba[0][i]*100:.1f}%")

    # -------------------- GAUGE CHART --------------------
    st.markdown("### 📌 Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max_prob * 100,
        title={'text': f"{risk_status}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "blue", 'width': 4},
                'thickness': 0.75,
                'value': max_prob * 100
            }
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # -------------------- EXPLANATION --------------------
    st.markdown("### 🔎 Key Factors Behind Prediction")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    try:
        shap_values_instance = shap_values[prediction_encoded[0]][0]
    except (IndexError, TypeError):
        try:
            shap_values_instance = shap_values[0, prediction_encoded[0], :]
        except Exception:
            shap_values_instance = None

    if shap_values_instance is not None:
        abs_vals = np.abs(shap_values_instance)
        percents = (abs_vals / np.sum(abs_vals)) * 100 if np.sum(abs_vals) > 0 else None

        if percents is not None:
            feature_pairs = sorted(zip(feature_names, percents), key=lambda x: x[1], reverse=True)
            labels = [f"{f} ({p:.1f}%)" for f, p in feature_pairs]
            values = [p for _, p in feature_pairs]

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
            fig.update_layout(title_text="Feature Contribution to Prediction")
            st.plotly_chart(fig, use_container_width=True)

    # -------------------- LIFESTYLE --------------------
    st.markdown("### 🧬 Patient Lifestyle Info (Not in Model)")
    st.info(f"""
    - Sex: **{sex}**
    - Tobacco Smoking: **{smoking}**
    - Alcohol Consumption: **{alcohol}**
    - Physical Activity: **{physical_activity}**
    - Family History of Diabetes: **{family_history}**
    """)

    # -------------------- HEALTH TIPS --------------------
    st.markdown("## 💡 Health Tips (WHO)")
    if "Diabetes" in risk_status:
        st.warning("""
        - Eat a balanced diet (fruits, vegetables, lean proteins).  
        - Stay active: 150 min/week moderate activity.  
        - Monitor sugar regularly & follow medication.  
        """)
    elif "Pre-Diabetes" in risk_status:
        st.warning("""
        - Reduce weight (5–10%).  
        - Avoid smoking, manage stress.  
        - Eat healthy & stay active.  
        """)
    else:
        st.success("""
        - Maintain your lifestyle: healthy eating, exercise, sleep.  
        - Regular check-ups for prevention.  
        """)

# -------------------- FOOTER --------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;'>✨ Built with ❤️ using <b>Streamlit, Scikit-learn, SHAP & Plotly</b></p>",
    unsafe_allow_html=True
)
