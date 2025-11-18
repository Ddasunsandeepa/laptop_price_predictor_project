import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.express as px

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# ================= CUSTOM SUPERB THEME =================
st.markdown("""
<style>
/* Main background and font */
.stApp {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #f0f6fc;
}

/* Headers */
h1, h2, h3 {
    color: #ffdd59;
    text-align: center;
    font-weight: 800;
    text-shadow: 2px 2px 8px #000000;
}

/* Inputs */
.stTextInput, .stSelectbox, .stSlider, .stNumberInput {
    color: #0f2027 !important;
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 0.8em 1.5em;
    font-weight: 700;
    font-size: 18px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}
.stButton > button:hover {
    transform: scale(1.1);
    box-shadow: 0px 6px 20px rgba(0,0,0,0.4);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f1c2c, #928dab);
    color: white;
    font-weight: bold;
}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
    transition: transform 0.3s ease-in-out;
}
.card:hover {
    transform: translateY(-5px) scale(1.02);
}

/* Pie Chart */
[data-testid="stPlotlyChart"] {
    border-radius: 20px;
    padding: 10px;
}

/* Divider */
hr {
    border: 1px solid #ffdd59;
    opacity: 0.3;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
MODEL_PATH = "../src/fc212030_Mudhitha/lapdata.pkl"
pipeline = joblib.load(MODEL_PATH)

EXCHANGE_API_URL = "https://api.exchangerate.host/convert?from=INR&to=LKR"
def get_inr_to_lkr_rate():
    try:
        response = requests.get(EXCHANGE_API_URL, timeout=5)
        return response.json().get("result", 4.0)
    except:
        return 4.0

# ================= PAGE HEADER =================
st.markdown("<h1>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ffd966;'>Predict laptop prices instantly with ML ✨</p>", unsafe_allow_html=True)
st.markdown("<hr>")

# ================= SIDEBAR =================
st.sidebar.markdown("### About This App")
st.sidebar.write("""
Predict laptop prices using a **trained ML model**.  
⚡ Modern UI with animations  
📊 Interactive component charts  
Responsive design for all screens
""")

# ================= INPUT FIELDS =================
st.markdown("### Enter Laptop Specifications")

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
    touchscreen = st.checkbox("TouchScreen")
with colx2:
    ips = st.checkbox("IPS Display")


# ================= PREDICT BUTTON =================
if st.button("Predict Laptop Price"):
    try:
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

        log_price_inr = pipeline.predict(input_df)[0]
        price_inr = np.exp(log_price_inr)
        rate = get_inr_to_lkr_rate()
        price_lkr = price_inr * rate

        # Display Prediction Card (LKR only)
        st.markdown(f"""
        <div class="card">
            <h2>💰 Predicted Price</h2>
            <h1>₨ {price_lkr:,.2f} LKR</h1>
        </div>
        """, unsafe_allow_html=True)

        # ================= COMPONENT DISTRIBUTION =================
        st.markdown("### Component Distribution")
        chart_data = pd.DataFrame({
            'Component': ['RAM', 'SSD', 'HDD', 'Weight', 'PPI'],
            'Value': [ram, ssd, hdd, weight, ppi]
        })
        colors = ['#ff6b6b', '#f7b32b', '#6bc1ff', '#9b59b6', '#22d69d']
        fig = px.pie(
            chart_data,
            names='Component',
            values='Value',
            hole=0.4,
            color='Component',
            color_discrete_sequence=colors
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color='white',
            title_font_color='#ffdd59',
            legend_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
