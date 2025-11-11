import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import os

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
)

# ================= CUSTOM DARK THEME CSS =================
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
}

/* Headings */
h1, h2, h3, h4 {
    color: #58a6ff;
    text-align: center;
    font-weight: 600;
}

/* Inputs and text */
label, .stTextInput, .stSelectbox, .stSlider, .stNumberInput {
    color: #e6edf3 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    font-weight: bold;
    transition: 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #2ea043, #3fb950);
    transform: scale(1.03);
}

/* Divider line */
hr {
    border: 1px solid #30363d;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Plot background */
.plotly-graph-div, .js-plotly-plot, .main-svg {
    background-color: #0d1117 !important;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
MODEL_PATH = "../src/fc212030_Mudhitha/lapdata.pkl"
pipeline = joblib.load(MODEL_PATH)

EXCHANGE_API_URL = "https://api.exchangerate.host/convert?from=INR&to=LKR"

def get_inr_to_lkr_rate():
    """Fetch INR to LKR exchange rate"""
    try:
        response = requests.get(EXCHANGE_API_URL, timeout=5)
        return response.json().get("result", 4.0)
    except:
        return 4.0

# ================= PAGE HEADER =================
st.markdown("<h1>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#8b949e;'>A modern AI-powered estimator built with Machine Learning and Explainable AI (SHAP)</p>", unsafe_allow_html=True)
st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.markdown("### 🧠 About This App")
st.sidebar.write("""
This application uses a trained **Machine Learning pipeline**  
to predict laptop prices and provide **Explainable AI** insights using **SHAP** values.
""")

st.sidebar.markdown("### ⚙️ Model Info")
st.sidebar.write("- Type: Regression Model\n- Framework: scikit-learn\n- XAI: SHAP Explainer")

# ================= INPUT FIELDS =================
st.markdown("### Input Specifications")

companies = ["Dell", "HP", "Lenovo", "Asus", "Apple", "Acer", "MSI"]
types = ["Ultrabook", "Gaming", "Notebook", "2 in 1", "Workstation"]
os_options = ["Windows", "MacOS", "Linux", "Chrome OS"]
cpu_options = ["Intel i3", "Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"]
gpu_options = ["Intel", "Nvidia GTX", "Nvidia RTX", "AMD Radeon"]

col1, col2, col3 = st.columns(3)

with col1:
    company = st.selectbox("Company", companies)
    typename = st.selectbox("Type", types)
    opsys = st.selectbox("Operating System", os_options)

with col2:
    cpu = st.selectbox("CPU", cpu_options)
    gpu = st.selectbox("GPU", gpu_options)
    ram = st.slider("RAM (GB)", 2, 64, 8, step=2)

with col3:
    ssd = st.slider("SSD (GB)", 0, 2048, 512, step=128)
    hdd = st.slider("HDD (GB)", 0, 2048, 0, step=128)
    weight = st.number_input("Weight (kg)", 0.8, 5.0, step=0.1)
    ppi = st.number_input("PPI (Display Density)", 60, 600, step=1)

colx1, colx2 = st.columns(2)
with colx1:
    touchscreen = st.checkbox("TouchScreen", value=False)
with colx2:
    ips = st.checkbox("IPS Display", value=False)

st.markdown("---")

# ================= PREDICT BUTTON =================
if st.button("Predict Laptop Price"):
    try:
        # Prepare input
        input_data = {
            "Company": company,
            "TypeName": typename,
            "Ram": ram,
            "OpSys": opsys,
            "Weight": weight,
            "CPU": cpu,
            "SSD": ssd,
            "HDD": hdd,
            "GPU": gpu,
            "TouchScreen": int(touchscreen),
            "IPS": int(ips),
            "PPI": ppi
        }

        input_df = pd.DataFrame([input_data])

        # Predict log price → convert to INR
        log_price_inr = pipeline.predict(input_df)[0]
        price_inr = np.exp(log_price_inr)
        rate = get_inr_to_lkr_rate()
        price_lkr = price_inr * rate

        # Display prediction
        st.markdown("<h2 style='text-align:center; color:#3fb950;'>Predicted Price</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center; color:#58a6ff;'>₹ {price_inr:,.2f} INR  ≈  ₨ {price_lkr:,.2f} LKR</h3>", unsafe_allow_html=True)
        st.markdown("---")

                # ================= VISUALIZATIONS =================
        st.markdown("### Component Distribution Overview")
        chart_data = pd.DataFrame({
            'Component': ['RAM', 'SSD', 'HDD', 'Weight', 'PPI'],
            'Value': [ram, ssd, hdd, weight, ppi]
        })

        # Define a color palette suitable for dark background
        colors = ['#58a6ff', '#3fb950', '#f85149', '#d29922', '#c084fc']

        fig = px.pie(
            chart_data,
            names='Component',
            values='Value',
            title='Laptop Component Distribution',
            hole=0.3,
            color='Component',
            color_discrete_sequence=colors
        )

        # Update layout for dark mode
        fig.update_layout(
            paper_bgcolor="#0d1117",  # background outside the chart
            plot_bgcolor="#0d1117",   # background inside chart
            font_color='white',
            title_font_color='#58a6ff',
            legend_font_color='white'
        )

        # Display chart
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


        # ================= EXPLAINABLE AI =================
        st.markdown("### Explainable AI (Feature Impact)")

        try:
            with st.spinner("Computing SHAP values..."):
                # Transform the input using pipeline preprocessing if exists
                if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
                    preprocessor = pipeline.named_steps['preprocessor']
                    model = pipeline.named_steps['model']
                    X_transformed = preprocessor.transform(input_df)
                    explainer = shap.Explainer(model, X_transformed)
                    shap_values = explainer(X_transformed)
                else:
                    # fallback if pipeline is simple
                    explainer = shap.Explainer(pipeline.predict, input_df)
                    shap_values = explainer(input_df)

                # Dark theme for SHAP bar plot
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10,5), facecolor='#0d1117')
                shap.plots.bar(shap_values, show=False)
                ax.set_facecolor('#0d1117')
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_color('white')
                st.pyplot(fig, transparent=True)

        except Exception as shap_err:
            st.warning(f"SHAP explanation unavailable: {shap_err}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
