import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Term Deposit Prediction",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "bank_marketing_final_model.pkl"

st.markdown("""
<style>
    :root {
        --sage: #A8B8A0;
        --sage-dark: #5F735C;
        --cream: #F8F5EE;
        --beige: #EEE8DC;
        --peach: #E8B9A8;
        --peach-dark: #A96F5C;
        --text: #3F463F;
        --muted: #747A73;
        --white: #FFFFFF;
    }

    .stApp {
        background: linear-gradient(180deg, #FAF8F3 0%, #F4F6F0 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 900px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        text-align: center;
        padding: 0.8rem 0 1.7rem 0;
    }

    .hero h1 {
        color: #465445;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }

    .hero p {
        color: var(--muted);
        font-size: 0.95rem;
        margin-top: 0;
    }

    .section-title {
        color: #526250;
        font-size: 1.22rem;
        font-weight: 700;
        margin: 1.15rem 0 0.75rem 0;
        letter-spacing: -0.01em;
    }

    div[data-testid="stForm"] {
        border: 1px solid #E4E6DE;
        border-radius: 18px;
        padding: 1.25rem 1.35rem 0.8rem 1.35rem;
        background: rgba(255,255,255,0.82);
        box-shadow: 0 8px 25px rgba(82, 98, 80, 0.06);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 9px !important;
        border-color: #D9DED4 !important;
        background-color: #FCFCF9 !important;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: #A8B8A0 !important;
    }

    div[data-testid="InputInstructions"],
    div[data-testid="stInputInstructions"] {
        display: none !important;
    }

    label[data-testid="stWidgetLabel"] p {
        color: #596356;
        font-weight: 550;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: #8FA58A;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.8rem;
        transition: 0.2s ease;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #72896D;
        color: white;
        border: none;
    }

    div[data-testid="stButton"] button {
        background: #EEE8DC;
        color: #596356;
        border: 1px solid #D9DED4;
        border-radius: 10px;
        font-weight: 600;
    }

    div[data-testid="stButton"] button:hover {
        background: #E3DDCF;
        color: #465445;
        border: 1px solid #C8D0C2;
    }

    .result-card {
        padding: 1.25rem 1.4rem;
        border-radius: 16px;
        border: 1px solid #E1E4DC;
        background: linear-gradient(135deg, #FFFFFF 0%, #F4F7F1 100%);
        box-shadow: 0 8px 22px rgba(82, 98, 80, 0.07);
        margin-top: 1rem;
    }

    .result-label {
        color: #7A8078;
        font-size: 0.82rem;
        margin-bottom: 0.25rem;
    }

    .result-value {
        font-size: 1.7rem;
        font-weight: 700;
    }

    .prediction-positive {
        color: #6C8B69;
    }

    .prediction-negative {
        color: #B17A62;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.65);
        border-radius: 12px;
        padding: 0.65rem 0.85rem;
        border: 1px solid #E5E7E0;
    }

    div[data-testid="stMetricLabel"] {
        color: #747A73 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #526250 !important;
    }

    .note {
        color: #7A8078;
        font-size: 0.78rem;
        line-height: 1.45;
        margin-top: 1rem;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

# IMPORTANT:
# form_version must be initialized BEFORE it is used anywhere.
if "form_version" not in st.session_state:
    st.session_state.form_version = 0


# ============================================================
# OPTIONS
# ============================================================

EDUCATION_MAPPING = {
    "basic.4y": "basic",
    "basic.6y": "basic",
    "basic.9y": "basic",
    "high.school": "high_school",
    "professional.course": "professional",
    "university.degree": "university",
    "illiterate": "illiterate",
    "unknown": "unknown"
}

JOB_OPTIONS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid",
    "management", "retired", "self-employed", "services",
    "student", "technician", "unemployed", "unknown"
]

MARITAL_OPTIONS = ["divorced", "married", "single", "unknown"]
DEFAULT_OPTIONS = ["no", "yes", "unknown"]
CONTACT_OPTIONS = ["cellular", "telephone"]
MONTH_OPTIONS = ["jan", "feb", "mar", "apr", "may", "jun",
                 "jul", "aug", "sep", "oct", "nov", "dec"]
DAY_OPTIONS = ["mon", "tue", "wed", "thu", "fri"]
POUTCOME_OPTIONS = ["failure", "nonexistent", "success"]


@st.cache_data
def load_economic_options():
    dataset_path = "bank-additional-full.csv"

    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "bank-additional-full.csv"
        )

    if not os.path.exists(dataset_path):
        return {
            "emp.var.rate": [
                -3.4, -2.9, -1.8, -1.7, -1.1, -0.5,
                -0.2, -0.1, 1.1, 1.4
            ],
            "cons.price.idx": [
                92.201, 92.379, 92.431, 92.649, 92.713,
                93.075, 93.444, 93.798, 93.994, 94.465,
                94.601, 94.767
            ],
            "cons.conf.idx": [
                -50.8, -49.5, -47.1, -46.2, -42.7, -41.8,
                -40.8, -40.3, -36.4, -34.8, -33.0, -26.9
            ],
            "euribor3m": [
                0.634, 0.635, 0.636, 0.637, 0.650,
                1.000, 2.000, 3.000, 4.000, 4.857, 5.045
            ],
            "nr.employed": [
                4963.6, 4991.6, 5008.7, 5017.5, 5023.5,
                5076.2, 5099.1, 5176.3, 5191.0, 5195.8, 5228.1
            ]
        }

    df = pd.read_csv(dataset_path, sep=";")

    columns = [
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed"
    ]

    return {
        col: sorted(df[col].dropna().unique().tolist())
        for col in columns
    }


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"File {MODEL_PATH} tidak ditemukan."

    try:
        return joblib.load(MODEL_PATH), None
    except Exception as e:
        return None, str(e)


# ============================================================
# LOAD MODEL / DATA
# ============================================================

artifact, error = load_model()
economic_options = load_economic_options()

st.markdown(
    '<div class="hero">'
    '<h1>🏦 Term Deposit Prediction</h1>'
    '<p>Gunakan insight data untuk membantu menentukan prioritas campaign</p>'
    '</div>',
    unsafe_allow_html=True
)

if artifact is None:
    st.error(error)
    st.stop()

model = artifact["model"]
threshold = float(artifact.get("threshold", 0.50))


# ============================================================
# CLEAR BUTTON
# ============================================================

clear_col1, clear_col2 = st.columns([5, 1])

with clear_col2:
    clear = st.button("↺ Clear", use_container_width=True)

if clear:
    # Change every widget key to a new version.
    # This forces Streamlit to create fresh widgets with empty values.
    st.session_state.form_version += 1
    st.rerun()


# ============================================================
# PREDICTION FORM
# ============================================================

form_key = f"prediction_form_{st.session_state.form_version}"

with st.form(form_key):

    st.markdown(
        '<div class="section-title">👤 Customer Profile</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "👤 Age",
            min_value=18,
            max_value=100,
            value=None,
            step=1,
            placeholder="Masukkan usia",
            key=f"{st.session_state.form_version}_age"
        )

        job = st.selectbox(
            "💼 Job",
            JOB_OPTIONS,
            index=None,
            placeholder="Pilih job...",
            key=f"{st.session_state.form_version}_job"
        )

    with c2:
        marital = st.selectbox(
            "💍 Marital",
            MARITAL_OPTIONS,
            index=None,
            placeholder="Pilih marital status...",
            key=f"{st.session_state.form_version}_marital"
        )

        education = st.selectbox(
            "🎓 Education",
            list(EDUCATION_MAPPING),
            index=None,
            placeholder="Pilih education...",
            key=f"{st.session_state.form_version}_education"
        )

    with c3:
        default = st.selectbox(
            "💳 Default",
            DEFAULT_OPTIONS,
            index=None,
            placeholder="Pilih...",
            key=f"{st.session_state.form_version}_default"
        )

        housing = st.selectbox(
            "🏠 Housing Loan",
            DEFAULT_OPTIONS,
            index=None,
            placeholder="Pilih...",
            key=f"{st.session_state.form_version}_housing"
        )

    loan = st.selectbox(
        "💰 Personal Loan",
        DEFAULT_OPTIONS,
        index=None,
        placeholder="Pilih...",
        key=f"{st.session_state.form_version}_loan"
    )

    st.markdown(
        '<div class="section-title">📣 Campaign</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        contact = st.selectbox(
            "📞 Contact",
            CONTACT_OPTIONS,
            index=None,
            placeholder="Pilih contact...",
            key=f"{st.session_state.form_version}_contact"
        )

        month = st.selectbox(
            "📅 Month",
            MONTH_OPTIONS,
            index=None,
            placeholder="Pilih month...",
            key=f"{st.session_state.form_version}_month"
        )

    with c2:
        day_of_week = st.selectbox(
            "🗓️ Day",
            DAY_OPTIONS,
            index=None,
            placeholder="Pilih day...",
            key=f"{st.session_state.form_version}_day_of_week"
        )

        campaign = st.number_input(
            "📣 Campaign Contacts",
            min_value=1,
            max_value=100,
            value=None,
            step=1,
            placeholder="Masukkan jumlah kontak",
            key=f"{st.session_state.form_version}_campaign"
        )

    with c3:
        pdays = st.number_input(
            "⏳ Days Since Previous Contact",
            min_value=0,
            max_value=999,
            value=None,
            step=1,
            placeholder="Masukkan hari",
            help="Gunakan 999 jika belum pernah dihubungi sebelumnya.",
            key=f"{st.session_state.form_version}_pdays"
        )

        previous = st.number_input(
            "🔄 Previous Contacts",
            min_value=0,
            max_value=20,
            value=None,
            step=1,
            placeholder="Masukkan jumlah kontak",
            key=f"{st.session_state.form_version}_previous"
        )

    poutcome = st.selectbox(
        "📊 Previous Campaign Outcome",
        POUTCOME_OPTIONS,
        index=None,
        placeholder="Pilih outcome...",
        key=f"{st.session_state.form_version}_poutcome"
    )

    st.markdown(
        '<div class="section-title">📈 Economic Context</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        emp_var_rate = st.selectbox(
            "📈 Employment Variation Rate",
            economic_options["emp.var.rate"],
            index=None,
            placeholder="Pilih nilai...",
            format_func=lambda x: f"{x:.3f}",
            key=f"{st.session_state.form_version}_emp_var_rate"
        )

        cons_price_idx = st.selectbox(
            "🛒 Consumer Price Index",
            economic_options["cons.price.idx"],
            index=None,
            placeholder="Pilih nilai...",
            format_func=lambda x: f"{x:.3f}",
            key=f"{st.session_state.form_version}_cons_price_idx"
        )

    with c2:
        cons_conf_idx = st.selectbox(
            "🧠 Consumer Confidence Index",
            economic_options["cons.conf.idx"],
            index=None,
            placeholder="Pilih nilai...",
            format_func=lambda x: f"{x:.3f}",
            key=f"{st.session_state.form_version}_cons_conf_idx"
        )

        euribor3m = st.selectbox(
            "📉 Euribor 3M",
            economic_options["euribor3m"],
            index=None,
            placeholder="Pilih nilai...",
            format_func=lambda x: f"{x:.3f}",
            key=f"{st.session_state.form_version}_euribor3m"
        )

    with c3:
        nr_employed = st.selectbox(
            "👥 Number of Employees",
            economic_options["nr.employed"],
            index=None,
            placeholder="Pilih nilai...",
            format_func=lambda x: f"{x:.1f}",
            key=f"{st.session_state.form_version}_nr_employed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🔍 Predict Subscription",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    required_fields = {
        "Age": age,
        "Job": job,
        "Marital": marital,
        "Education": education,
        "Default": default,
        "Housing Loan": housing,
        "Personal Loan": loan,
        "Contact": contact,
        "Month": month,
        "Day": day_of_week,
        "Campaign Contacts": campaign,
        "Days Since Previous Contact": pdays,
        "Previous Contacts": previous,
        "Previous Campaign Outcome": poutcome,
        "Employment Variation Rate": emp_var_rate,
        "Consumer Price Index": cons_price_idx,
        "Consumer Confidence Index": cons_conf_idx,
        "Euribor 3M": euribor3m,
        "Number of Employees": nr_employed,
    }

    missing_fields = [
        name
        for name, value in required_fields.items()
        if value is None or (isinstance(value, str) and not value.strip())
    ]

    if missing_fields:
        st.warning(
            "Mohon lengkapi semua kolom sebelum melakukan prediction. "
            f"Kolom yang belum diisi: {', '.join(missing_fields)}"
        )
        st.stop()

    if age <= 25:
        age_group = "<=25"
    elif age <= 35:
        age_group = "26-35"
    elif age <= 45:
        age_group = "36-45"
    elif age <= 55:
        age_group = "46-55"
    elif age <= 65:
        age_group = "56-65"
    else:
        age_group = ">65"

    if campaign <= 1:
        campaign_group = "1"
    elif campaign <= 2:
        campaign_group = "2"
    elif campaign <= 3:
        campaign_group = "3"
    elif campaign <= 5:
        campaign_group = "4-5"
    elif campaign <= 10:
        campaign_group = "6-10"
    else:
        campaign_group = ">10"

    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "age_group": age_group,
        "campaign_group": campaign_group,
        "education_group": EDUCATION_MAPPING[education],
        "previously_contacted": 0 if pdays == 999 else 1
    }])

    try:
        probability = float(
            model.predict_proba(input_data)[:, 1][0]
        )

        prediction = int(probability >= threshold)

        st.markdown("---")

        result = "Subscription" if prediction else "No Subscription"
        css = "positive" if prediction else "negative"

        st.markdown(
            f'<div class="section-title">Prediction Result</div>'
            f'<div class="result-card">'
            f'<div class="result-label">Prediction</div>'
            f'<div class="result-value {css}">{result}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.metric("Probability", f"{probability:.2%}")

        if prediction:
            st.success(
                "Nasabah diprediksi memiliki kemungkinan subscription "
                "berdasarkan hasil model."
            )
        else:
            st.info(
                "Nasabah diprediksi tidak melakukan subscription "
                "berdasarkan hasil model."
            )

        st.markdown(
            '<div class="note">'
            'Catatan: probabilitas merupakan output model, bukan jaminan '
            'bahwa nasabah akan melakukan subscription. '
            'F2-Score digunakan sebagai primary metric pada project.'
            '</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")


st.markdown(
    '<div class="note" style="text-align:center">'
    'Bank Marketing Project · XGBoost - Balanced'
    '</div>',
    unsafe_allow_html=True
)
