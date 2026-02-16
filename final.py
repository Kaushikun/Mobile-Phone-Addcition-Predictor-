# 🚀 FINAL VERSION - CENTERED TITLE + BOTTOM FOOTER
# Copy ALL into 1 Colab cell → Run → Click URL

!pip install streamlit pyngrok plotly pandas numpy scikit-learn -q --no-cache-dir


!pkill -f streamlit || true &>/dev/null
!pkill -f ngrok || true &>/dev/null
from pyngrok import ngrok; ngrok.kill()


app_code = '''import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = []
if "model" not in st.session_state:
    st.session_state.model = None

st.set_page_config(page_title="Mobile Addiction Predictor", layout="wide", page_icon="📱")


st.markdown("""
    <div style='text-align: center; font-size: 3.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 2rem;'>
        📱 Mobile Phone Addiction Predictor
    </div>
""", unsafe_allow_html=True)



@st.cache_data
def train_ml_model():
    np.random.seed(42)
    n_samples = 500

    data = {
        'daily_hours': np.random.normal(5, 2, n_samples).clip(0, 24),
        'checks_per_hour': np.random.normal(12, 8, n_samples).clip(0, 50),
        'notifications': np.random.normal(60, 40, n_samples).clip(0, 500),
        'failed_attempts': np.random.poisson(1, n_samples).clip(0, 10),
        'age': np.random.normal(25, 8, n_samples).clip(16, 60),
        'bedtime_use': np.random.choice([0,1,2,3], n_samples, p=[0.2,0.3,0.3,0.2]),
        'anxiety': np.random.choice([0,1,2,3], n_samples, p=[0.3,0.3,0.25,0.15]),
        'sleep_impact': np.random.choice([0,1,2,3], n_samples, p=[0.25,0.35,0.25,0.15]),
        'work_impact': np.random.choice([0,1,2,3], n_samples, p=[0.4,0.3,0.2,0.1]),
        'avoidance': np.random.choice([0,1,2,3], n_samples, p=[0.3,0.35,0.25,0.1])
    }

    df = pd.DataFrame(data)
    df['addiction_prob'] = (
        0.3*df.daily_hours/24 + 0.15*df.checks_per_hour/50 + 0.1*df.notifications/500 +
        0.2*df.failed_attempts/10 + 0.05*(30-df.age)/44 +
        0.1*df.bedtime_use/3 + 0.15*df.anxiety/3 + 0.15*df.sleep_impact/3 +
        0.15*df.work_impact/3 + 0.1*df.avoidance/3
    )
    df['addicted'] = (df.addiction_prob > 0.5).astype(int)

    features = ['daily_hours', 'checks_per_hour', 'notifications', 'failed_attempts',
                'age', 'bedtime_use', 'anxiety', 'sleep_impact', 'work_impact', 'avoidance']

    X = df[features]
    y = df['addicted']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


if st.session_state.model is None:
    st.session_state.model = train_ml_model()

model = st.session_state.model


st.header("📋 Answer 10 Questions")
with st.form(key="quiz", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        q1 = st.slider("1. Daily phone hours?", 0, 24, 4)
        q2 = st.slider("2. Phone checks/hour?", 0, 50, 10)
        q3 = st.selectbox("3. Bedtime phone use?", ["Never", "Sometimes", "Often", "Always"], key="q3")

    with col2:
        q4 = st.selectbox("4. Anxious without phone?", ["Never", "Sometimes", "Often", "Always"], key="q4")
        q5 = st.slider("5. Daily notifications?", 0, 500, 50)
        q6 = st.selectbox("6. Avoid boredom with phone?", ["Never", "Sometimes", "Often", "Always"], key="q6")

    col3, col4 = st.columns(2)
    with col3:
        q7 = st.selectbox("7. Sleep affected?", ["Never", "Sometimes", "Often", "Always"], key="q7")
        q8 = st.selectbox("8. Work/study affected?", ["Never", "Sometimes", "Often", "Always"], key="q8")

    with col4:
        q9 = st.slider("9. Failed to reduce usage?", 0, 10, 1)
        q10 = st.selectbox("10. Others notice your usage?", ["Never", "Sometimes", "Often", "Always"], key="q10")
        age = st.number_input("Age", 16, 60, 22)

    submit = st.form_submit_button("Calculate Risk", type="primary")


if submit:
    st.header("📊 RESULTS")

    cat_map = {"Never": 0, "Sometimes": 1, "Often": 2, "Always": 3}
    features = np.array([[q1, q2, q5, q9, age, cat_map[q3], cat_map[q4],
                         cat_map[q7], cat_map[q8], cat_map[q6]]])

    ml_prob = model.predict_proba(features)[0, 1]
    prediction = "ADDICTED" if ml_prob > 0.5 else "NOT ADDICTED"

    
    if prediction == "ADDICTED":
        st.error(f"🚨 **{prediction}** (Risk: {ml_prob*100:.0f}%)")
    else:
        st.success(f"✅ **{prediction}** (Risk: {ml_prob*100:.0f}%)")

    
    result = {"age": age, "risk": ml_prob*100, "daily_hours": q1, "status": prediction}
    st.session_state.results.append(result)

    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Daily Usage", f"{q1} hours")
    with col2: st.metric("Checks/Hour", f"{q2}")
    with col3: st.metric("Age", f"{age}y")

  
    fig = px.line_polar(
        pd.DataFrame({
            "Level": ["Safe", "Warning", "Danger", "YOUR SCORE"],
            "Risk": [30, 60, 100, ml_prob*100]
        }),
        r="Risk", theta="Level",
        color_discrete_sequence=["green", "orange", "red", "purple"]
    )
    st.plotly_chart(fig, use_container_width=True)

    
    csv = pd.DataFrame(st.session_state.results).to_csv(index=False)
    st.download_button("📥 Download Results", csv, "results.csv", "text/csv")


if st.session_state.results:
    st.subheader("📈 Community Stats")
    col1, col2 = st.columns(2)
    with col1: st.metric("Total Tests", len(st.session_state.results))
    with col2: st.metric("Avg Risk", f"{np.mean([r['risk'] for r in st.session_state.results]):.0f}%")


st.markdown("---")
st.header("🆘 How to Get Away From Addiction")
st.info("🚀 **Watch this video for proven strategies to break phone addiction!**")



st.video("https://www.youtube.com/watch?v=ptrbG6675JA")


st.markdown("""
    <div style='text-align: center; padding: 3rem 0; font-size: 1.2rem; color: #666;'>
        <hr style='border: 1px solid #ddd; margin: 2rem 0;'>
        *Mobile Phone Addiction Predictor | BY : Kaushikun Krishna Kumar*
    </div>
""", unsafe_allow_html=True)
'''

#Running the web
with open("app.py", "w") as f: f.write(app_code)
!nohup streamlit run app.py --server.port 8501 --server.headless true >/dev/null 2>&1 &
import time; time.sleep(3)
public_url = ngrok.connect(8501)
print(f"🚀 LIVE WITH RECOVERY VIDEO: {public_url}")
