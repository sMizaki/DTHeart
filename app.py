import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# 🎨 ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Custom CSS สำหรับความสวยงาม
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .title {
        text-align: center;
        color: #2c3e50;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        margin: 20px 0;
    }
    .safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .danger {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 25px;
        font-size: 1.1em;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# 📂 โหลดโมเดล
@st.cache_resource
def load_model():
    with open('heart_disease_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

try:
    model_data = load_model()
    model = model_data['model']
    feature_names = model_data['feature_names']
    model_accuracy = model_data['accuracy']
except:
    st.error("❌ ไม่พบไฟล์ heart_disease_model.pkl กรุณาวางไฟล์ในโฟลเดอร์เดียวกัน")
    st.stop()

# 🏥 Header
st.markdown('<p class="title">🫀 Heart Disease Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ระบบทำนายความเสี่ยงโรคหัวใจด้วย Machine Learning</p>', unsafe_allow_html=True)

# 📊 Sidebar แสดงข้อมูลโมเดล
with st.sidebar:
    st.header("📊 ข้อมูลโมเดล")
    st.metric("Algorithm", "Decision Tree")
    st.metric("Accuracy", f"{model_accuracy*100:.2f}%")
    st.metric("Features", len(feature_names))
    
    st.markdown("---")
    st.markdown("### 📖 คำแนะนำ")
    st.info("""
    กรอกข้อมูลสุขภาพของคุณในฟอร์มด้านล่าง 
    ระบบจะวิเคราะห์และทำนายความเสี่ยงโรคหัวใจ
    
    ⚠️ **คำเตือน**: ผลลัพธ์เป็นเพียงการคาดการณ์ 
    ไม่ใช่การวินิจฉัยทางการแพทย์
    """)

# 📝 Input Form
st.markdown("### 📋 กรอกข้อมูลสุขภาพ")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("👤 อายุ (ปี)", 20, 90, 50)
    sex = st.selectbox("⚧ เพศ", ["ชาย", "หญิง"])
    sex_val = 1 if sex == "ชาย" else 0
    chest_pain = st.selectbox(
        "💔 ประเภทอาการเจ็บหน้าอก",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Typical Angina",
            2: "Atypical Angina",
            3: "Non-Anginal Pain",
            4: "Asymptomatic"
        }[x]
    )

with col2:
    resting_bp = st.number_input("🩸 ความดันโลหิตขณะพัก (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("🧪 โคเลสเตอรอล (mg/dl)", 0, 600, 200)
    fasting_bs = st.selectbox(
        "🍬 น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl?",
        ["ไม่ใช่", "ใช่"]
    )
    fasting_bs_val = 1 if fasting_bs == "ใช่" else 0

with col3:
    resting_ecg = st.selectbox(
        "📈 ผล ECG ขณะพัก",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T wave abnormality",
            2: "Left ventricular hypertrophy"
        }[x]
    )
    max_hr = st.number_input("💓 อัตราการเต้นหัวใจสูงสุด", 50, 220, 150)
    exercise_angina = st.selectbox("🏃 เจ็บหน้าอกขณะออกกำลังกาย?", ["ไม่ใช่", "ใช่"])
    exercise_angina_val = 1 if exercise_angina == "ใช่" else 0

col4, col5 = st.columns(2)
with col4:
    oldpeak = st.number_input("📉 ST depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
with col5:
    st_slope = st.selectbox(
        "📐 ST Slope",
        [1, 2, 3],
        format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x]
    )

# 🔮 Predict Button
st.markdown("---")
if st.button("🔮 ทำนายผล", use_container_width=True):
    # สร้าง DataFrame จากข้อมูล input
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_val],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs_val],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_val],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })
    
    # ทำนาย
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # แสดงผล
    st.markdown("---")
    if prediction == 0:
        st.markdown('<div class="result-box safe">✅ ไม่มีความเสี่ยงโรคหัวใจ</div>', 
                   unsafe_allow_html=True)
        st.success(f"ความมั่นใจ: {probability[0]*100:.2f}%")
    else:
        st.markdown('<div class="result-box danger">⚠️ มีความเสี่ยงโรคหัวใจ</div>', 
                   unsafe_allow_html=True)
        st.error(f"ความมั่นใจ: {probability[1]*100:.2f}%")
    
    # แสดง Probability Gauge
    col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
    with col_g2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            title={'text': "ความเสี่ยง (%)", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#38ef7d"},
                    {'range': [30, 70], 'color': "#f39c12"},
                    {'range': [70, 100], 'color': "#eb3349"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # แสดงข้อมูล input ที่กรอก
    with st.expander("📋 ดูข้อมูลที่คุณกรอก"):
        st.dataframe(input_data.T.rename(columns={0: 'ค่า'}))

# 📚 Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🤖 พัฒนาด้วย Machine Learning (Decision Tree) + Streamlit</p>
    <p>⚕️ ผลลัพธ์เป็นเพียงการคาดการณ์ ควรปรึกษาแพทย์เพื่อวินิจฉัยที่ถูกต้อง</p>
</div>
""", unsafe_allow_html=True)