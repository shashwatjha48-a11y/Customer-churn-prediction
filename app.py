import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("Models/churn_pipeline.pkl")


model = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL APPLICATION
       ====================================================== */

    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    .main {
        background-color: #000000 !important;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 35px;
        padding-bottom: 50px;
    }


    /* ======================================================
       HEADER
       ====================================================== */

    .netflix-logo {
        color: #E50914;
        font-size: 52px;
        font-weight: 900;
        letter-spacing: -3px;
        margin-bottom: 0px;
    }

    .project-title {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        margin-top: 0px;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #999999;
        font-size: 17px;
        margin-top: 0px;
        margin-bottom: 35px;
    }


    /* ======================================================
       SECTION HEADERS
       ====================================================== */

    .section-header {
        color: #ffffff;
        font-size: 23px;
        font-weight: 700;
        border-left: 4px solid #E50914;
        padding-left: 12px;
        margin-top: 30px;
        margin-bottom: 20px;
    }


    /* ======================================================
       INPUT LABELS
       ====================================================== */

    label {
        color: #dddddd !important;
        font-weight: 500 !important;
    }


    /* ======================================================
       NUMBER INPUTS
       ====================================================== */

    div[data-baseweb="input"] {
        background-color: #181818 !important;
        border: 1px solid #333333 !important;
        border-radius: 5px !important;
    }

    div[data-baseweb="input"]:focus-within {
        border: 1px solid #E50914 !important;
    }

    input {
        color: #ffffff !important;
        background-color: #181818 !important;
    }


    /* ======================================================
       SELECT BOXES
       ====================================================== */

    div[data-baseweb="select"] > div {
        background-color: #181818 !important;
        border: 1px solid #333333 !important;
        color: #ffffff !important;
        border-radius: 5px !important;
    }


    /* ======================================================
       PREDICTION BUTTON
       ====================================================== */

    div.stButton > button {
        width: 100%;
        background-color: #E50914 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 5px !important;
        height: 52px;
        font-size: 17px;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        background-color: #B20710 !important;
        color: #ffffff !important;
        border: none !important;
    }


    /* ======================================================
       PREDICTION CARD
       ====================================================== */

    .prediction-card {
        background: #181818 !important;
        border: 1px solid #333333 !important;
        border-top: 4px solid #E50914 !important;
        border-radius: 8px;
        padding: 35px;
        margin-top: 30px;
        text-align: center;
        color: #ffffff !important;
    }

    .prediction-title {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 10px;
    }

    .probability {
        font-size: 52px;
        font-weight: 800;
        color: #E50914 !important;
        margin: 10px 0;
    }

    .prediction-description {
        color: #999999 !important;
        font-size: 15px;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #555555;
        font-size: 13px;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #1a1a1a;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="netflix-logo">NETFLIX</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="project-title">Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning powered customer retention analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.markdown(
    '<div class="section-header">Customer Information</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    region = st.selectbox(
        "Region",
        [
            "Africa",
            "Asia",
            "Europe",
            "North America",
            "Oceania",
            "South America"
        ]
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    subscription_type = st.selectbox(
        "Subscription Type",
        [
            "Basic",
            "Standard",
            "Premium"
        ]
    )

    device = st.selectbox(
        "Primary Device",
        [
            "TV",
            "Mobile",
            "Laptop",
            "Desktop",
            "Tablet"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Gift Card",
            "Crypto",
            "Debit Card",
            "PayPal",
            "Credit Card"
        ]
    )


# ------------------------------------------------------------
# COLUMN 3
# ------------------------------------------------------------

with col3:

    monthly_fee = st.number_input(
        "Monthly Fee",
        min_value=0.0,
        value=15.0,
        step=0.5
    )

    number_of_profiles = st.number_input(
        "Number of Profiles",
        min_value=1,
        max_value=10,
        value=2
    )

    favorite_genre = st.selectbox(
        "Favorite Genre",
        [
            "Action",
            "Comedy",
            "Documentary",
            "Drama",
            "Horror",
            "Romance",
            "Sci-Fi"
        ]
    )


# ============================================================
# VIEWING & ENGAGEMENT
# ============================================================

st.markdown(
    '<div class="section-header">Viewing & Engagement</div>',
    unsafe_allow_html=True
)


col4, col5, col6 = st.columns(3)


# ------------------------------------------------------------
# WATCH HOURS
# ------------------------------------------------------------

with col4:

    watch_hours = st.number_input(
        "Watch Hours",
        min_value=0.0,
        value=20.0,
        step=1.0
    )


# ------------------------------------------------------------
# LAST LOGIN
# ------------------------------------------------------------

with col5:

    last_login_days = st.number_input(
        "Days Since Last Login",
        min_value=0,
        value=5,
        step=1
    )


# ------------------------------------------------------------
# AVERAGE WATCH TIME
# ------------------------------------------------------------

with col6:

    avg_watch_time_per_day = st.number_input(
        "Average Watch Time Per Day",
        min_value=0.0,
        value=2.0,
        step=0.1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "PREDICT CUSTOMER CHURN"
)


# ============================================================
# PREDICTION LOGIC
# ============================================================

if predict_button:

    # Create dataframe using exactly the same
    # feature names used during model training.

    input_data = pd.DataFrame({

        "age": [age],

        "gender": [gender],

        "subscription_type": [subscription_type],

        "watch_hours": [watch_hours],

        "last_login_days": [last_login_days],

        "region": [region],

        "device": [device],

        "monthly_fee": [monthly_fee],

        "payment_method": [payment_method],

        "number_of_profiles": [number_of_profiles],

        "avg_watch_time_per_day": [
            avg_watch_time_per_day
        ],

        "favorite_genre": [favorite_genre]

    })


    # Make prediction

    prediction = model.predict(
        input_data
    )[0]


    # Get probability of churn

    probability = model.predict_proba(
        input_data
    )[0][1]


    # ========================================================
    # HIGH CHURN RESULT
    # ========================================================

    if prediction == 1:

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-title">
                    ⚠️ HIGH CHURN RISK
                </div>
                <div class="probability">
                    {probability:.1%}
                </div>
                <div class="prediction-description">
                    Estimated probability that this customer will churn.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # LOW CHURN RESULT
    # ========================================================

    else:

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-title">
                    ✓ LOW CHURN RISK
                </div>
                <div class="probability">
                    {probability:.1%}
                </div>
                <div class="prediction-description">
                    Estimated probability that this customer will churn.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    Netflix Customer Churn Prediction
    &nbsp;•&nbsp;
    Random Forest
    &nbsp;•&nbsp;
    Machine Learning
    &nbsp;•&nbsp;
    Streamlit
    <br><br>
    Built by <strong>Shashwat Jha</strong>
    </div>
    """,
    unsafe_allow_html=True
)
