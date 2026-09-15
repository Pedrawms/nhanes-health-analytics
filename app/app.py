import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "logistic_regression_final.joblib"
CONFIG_PATH = MODEL_DIR / "model_config.json"


# ============================================================
# Load Model and Configuration
# ============================================================

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

threshold = config["threshold"]


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Personal Health Analytics",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("🩺 Personal Health Analytics")

    st.write(
        "An educational machine learning application "
        "built using NHANES data."
    )

    st.divider()

    st.subheader("📌 Project Information")

    st.write(
        "**Dataset:** NHANES 2021–2023"
    )

    st.write(
        "**Model:** Logistic Regression"
    )

    st.write(
        "**Target:** Elevated Blood Pressure"
    )

    st.write(
        "**Decision Threshold:** "
        f"{threshold:.0%}"
    )

    st.divider()

    st.subheader("⚠️ Disclaimer")

    st.caption(
        "This application is intended for educational "
        "and research purposes only. It is not a medical "
        "diagnosis and should not replace professional "
        "medical advice."
    )


# ============================================================
# Header
# ============================================================

st.title("🩺 Personal Health Analytics")

st.write(
    "An educational machine learning tool for estimating "
    "the likelihood of elevated blood pressure."
)

st.info(
    "This tool is for educational and research purposes only. "
    "It is not a medical diagnosis and should not replace "
    "professional medical advice."
)


# ============================================================
# Personal Information
# ============================================================

st.header("👤 Personal Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30,
        step=1
    )

with col2:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

with col3:

    household_size = st.number_input(
        "Household Size",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )


# ============================================================
# Demographics
# ============================================================

st.header("🌎 Demographics")

col1, col2 = st.columns(2)

with col1:

    race = st.selectbox(
        "Race / Ethnicity",
        [
            "Mexican American",
            "Other Hispanic",
            "Non-Hispanic White",
            "Non-Hispanic Black",
            "Non-Hispanic Asian",
            "Other Race / Multiracial"
        ]
    )

with col2:

    education = st.selectbox(
        "Education Level",
        [
            "Less than 9th grade",
            "9th-11th grade",
            "High school graduate / GED",
            "Some college / AA",
            "College graduate or above"
        ]
    )


marital_status = st.selectbox(
    "Marital Status",
    [
        "Married / Living with partner",
        "Widowed / Divorced / Separated",
        "Never married"
    ]
)


# ============================================================
# Body Measurements
# ============================================================

st.header("⚖️ Body Measurements")

col1, col2, col3 = st.columns(3)

with col1:

    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=250.0,
        value=70.0,
        step=0.5
    )

with col2:

    height = st.number_input(
        "Height (cm)",
        min_value=120.0,
        max_value=220.0,
        value=170.0,
        step=0.5
    )

with col3:

    waist = st.number_input(
        "Waist Circumference (cm)",
        min_value=40.0,
        max_value=180.0,
        value=85.0,
        step=0.5
    )


# ============================================================
# BMI Calculation
# ============================================================

height_m = height / 100

bmi = weight / (height_m ** 2)

st.metric(
    "Calculated BMI",
    f"{bmi:.1f}"
)


# ============================================================
# Laboratory Measurements
# ============================================================

st.header("🧪 Laboratory Measurements")

col1, col2 = st.columns(2)

with col1:

    hdl = st.number_input(
        "HDL Cholesterol (mg/dL)",
        min_value=10.0,
        max_value=200.0,
        value=50.0,
        step=1.0
    )

with col2:

    vitamin_d = st.number_input(
        "Vitamin D (nmol/L)",
        min_value=5.0,
        max_value=500.0,
        value=75.0,
        step=1.0
    )


# ============================================================
# Smoking
# ============================================================

st.header("🚬 Smoking")

smoked_100 = st.radio(
    "Have you smoked at least 100 cigarettes in your lifetime?",
    [
        "Yes",
        "No"
    ],
    horizontal=True
)


# ============================================================
# Category Mappings
# ============================================================

gender_map = {
    "Male": 1,
    "Female": 2
}


race_map = {
    "Mexican American": 1,
    "Other Hispanic": 2,
    "Non-Hispanic White": 3,
    "Non-Hispanic Black": 4,
    "Non-Hispanic Asian": 6,
    "Other Race / Multiracial": 7
}


education_map = {
    "Less than 9th grade": 1,
    "9th-11th grade": 2,
    "High school graduate / GED": 3,
    "Some college / AA": 4,
    "College graduate or above": 5
}


marital_map = {
    "Married / Living with partner": 1,
    "Widowed / Divorced / Separated": 2,
    "Never married": 3
}


smoking_map = {
    "Yes": 1,
    "No": 2
}


# ============================================================
# Prediction
# ============================================================

st.divider()

if st.button(
    "🔍 Estimate Risk",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Create Input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "RIDAGEYR": age,

        "RIAGENDR": gender_map[gender],

        "RIDRETH3": race_map[race],

        "DMDEDUC2": education_map[education],

        "DMDMARTZ": marital_map[marital_status],

        "DMDHHSIZ": household_size,

        "BMXBMI": bmi,

        "BMXWAIST": waist,

        "BMXWT": weight,

        "BMXHT": height,

        "SMQ020": smoking_map[smoked_100],

        "LBDHDD": hdl,

        "LBXVIDMS": vitamin_d

    }])


    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    model_score = model.predict_proba(
        input_data
    )[0, 1]


    prediction = int(
        model_score >= threshold
    )


    # ========================================================
    # Prediction Result
    # ========================================================

    st.divider()

    st.header("📊 Prediction Result")


    # --------------------------------------------------------
    # Model Score
    # --------------------------------------------------------

    st.metric(
        "Model Score",
        f"{model_score:.1%}"
    )


    # --------------------------------------------------------
    # Progress Bar
    # --------------------------------------------------------

    st.progress(
        float(model_score)
    )


    # --------------------------------------------------------
    # Decision Threshold
    # --------------------------------------------------------

    st.caption(
        f"Decision threshold: {threshold:.0%}"
    )


    # ========================================================
    # Classification
    # ========================================================

    if prediction == 1:

        st.warning(
            "### Higher likelihood of elevated blood pressure"
        )

        st.write(
            "The model score is above the decision threshold "
            "used for classification."
        )

    else:

        st.success(
            "### Lower likelihood of elevated blood pressure"
        )

        st.write(
            "The model score is below the decision threshold "
            "used for classification."
        )


    # ========================================================
    # Interpretation
    # ========================================================

    with st.expander(
        "ℹ️ How should I interpret this result?"
    ):

        st.write(
            """
            The model score represents the output of a machine
            learning classification model.

            It should not be interpreted as a clinically calibrated
            probability or as a medical diagnosis.

            A score at or above the decision threshold is classified
            as a higher likelihood of elevated blood pressure by
            this model.

            A score below the threshold is classified as a lower
            likelihood.

            The model was developed using publicly available NHANES
            data and is intended for educational and research purposes.
            """
        )


    # ========================================================
    # Input Summary
    # ========================================================

    with st.expander(
        "🔎 View input summary"
    ):

        summary = pd.DataFrame({

            "Feature": [

                "Age",

                "Gender",

                "Race / Ethnicity",

                "Education",

                "Marital Status",

                "Household Size",

                "BMI",

                "Waist Circumference",

                "Weight",

                "Height",

                "HDL",

                "Vitamin D",

                "Smoked 100+ cigarettes"

            ],

            "Value": [

                age,

                gender,

                race,

                education,

                marital_status,

                household_size,

                f"{bmi:.1f}",

                f"{waist:.1f} cm",

                f"{weight:.1f} kg",

                f"{height:.1f} cm",

                f"{hdl:.1f} mg/dL",

                f"{vitamin_d:.1f} nmol/L",

                smoked_100

            ]

        })


        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )