!pip install streamlit pyngrok plotly pandas numpy scikit-learn -q --no-cache-dir
!pkill -f streamlit || true &>/dev/null
!pkill -f ngrok || true &>/dev/null
from pyngrok import ngrok; ngrok.kill()

app_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Mobile Addiction Predictor", layout="wide", page_icon="📱")
if "attemptStore" not in st.session_state:
    st.session_state.attemptStore = []
if "myModelObj" not in st.session_state:
    st.session_state.myModelObj = None
st.markdown("<h1 style='text-align:center;color:#1f77b4;'>📱 Mobile Phone Addiction Predictor</h1>", unsafe_allow_html=True)

@st.cache_data
def createModelRough():
   
    np.random.seed(42)
    totalSamples = 500

    fakeSurvey = {
        "daily_hours": np.random.normal(5,2,totalSamples).clip(0,24),
        "checks_hour": np.random.normal(12,8,totalSamples).clip(0,50),
        "notifications": np.random.normal(60,40,totalSamples).clip(0,500),
        "failed_try": np.random.poisson(1,totalSamples).clip(0,10),
        "age_user": np.random.normal(25,8,totalSamples).clip(16,60),
        "bed_use": np.random.choice([0,1,2,3], totalSamples, p=[0.2,0.3,0.3,0.2]),
        "anx_level": np.random.choice([0,1,2,3], totalSamples, p=[0.3,0.3,0.25,0.15]),
        "sleep_effect": np.random.choice([0,1,2,3], totalSamples, p=[0.25,0.35,0.25,0.15]),
        "work_effect": np.random.choice([0,1,2,3], totalSamples, p=[0.4,0.3,0.2,0.1]),
        "avoid_level": np.random.choice([0,1,2,3], totalSamples, p=[0.3,0.35,0.25,0.1])
    }

    df_local = pd.DataFrame(fakeSurvey)

    
    df_local["tempProb"] = (
        0.3*df_local.daily_hours/24 +
        0.15*df_local.checks_hour/50 +
        0.1*df_local.notifications/500 +
        0.2*df_local.failed_try/10 +
        0.05*(30-df_local.age_user)/44 +
        0.1*df_local.bed_use/3 +
        0.15*df_local.anx_level/3 +
        0.15*df_local.sleep_effect/3 +
        0.15*df_local.work_effect/3 +
        0.1*df_local.avoid_level/3
    )

    df_local["isAddicted"] = 0
    for i in range(len(df_local)):
        if df_local.loc[i,"tempProb"] > 0.5:
            df_local.loc[i,"isAddicted"] = 1
        else:
            df_local.loc[i,"isAddicted"] = 0   # explicit else (not needed actually)

    featureNames = ["daily_hours","checks_hour","notifications","failed_try",
                    "age_user","bed_use","anx_level","sleep_effect",
                    "work_effect","avoid_level"]

    Xvals = df_local[featureNames]
    Yvals = df_local["isAddicted"]

    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(Xvals, Yvals)

    return rf_temp


if st.session_state.myModelObj is None:
    st.session_state.myModelObj = createModelRough()

myModelObj = st.session_state.myModelObj


st.header("📋 Answer the Questions Honestly")

with st.form("quizForm"):

    colA, colB = st.columns(2)

    with colA:
        hoursUser = st.slider("1. Daily phone hours?",0,24,4)
        checkUser = st.slider("2. Phone checks/hour?",0,50,10)
        bedUser = st.selectbox("3. Bedtime phone use?",["Never","Sometimes","Often","Always"])

    with colB:
        anxUser = st.selectbox("4. Anxious without phone?",["Never","Sometimes","Often","Always"])
        notifUser = st.slider("5. Daily notifications?",0,500,50)
        avoidUser = st.selectbox("6. Avoid boredom with phone?",["Never","Sometimes","Often","Always"])

    colC, colD = st.columns(2)

    with colC:
        sleepUser = st.selectbox("7. Sleep affected?",["Never","Sometimes","Often","Always"])
        workUser = st.selectbox("8. Work/study affected?",["Never","Sometimes","Often","Always"])

    with colD:
        failUser = st.slider("9. Failed to reduce usage?",0,10,1)
        noticeUser = st.selectbox("10. Others notice your usage?",["Never","Sometimes","Often","Always"])
        ageInput = st.number_input("Age",16,60,22)

    submitNow = st.form_submit_button("Calculate Risk")


if submitNow:

    st.header("📊 Prediction Result")

    # mapping text to numeric manually
    mapVal = {"Never":0,"Sometimes":1,"Often":2,"Always":3}

    inputArr = []
    inputArr.append(hoursUser)
    inputArr.append(checkUser)
    inputArr.append(notifUser)
    inputArr.append(failUser)
    inputArr.append(ageInput)
    inputArr.append(mapVal[bedUser])
    inputArr.append(mapVal[anxUser])
    inputArr.append(mapVal[sleepUser])
    inputArr.append(mapVal[workUser])
    inputArr.append(mapVal[avoidUser])

    finalInput = np.array([inputArr])

    probValue = myModelObj.predict_proba(finalInput)[0][1]

    # extra safe clamp (rare case)
    if probValue < 0:
        probValue = 0
    if probValue > 1:
        probValue = 1

    resultLabel = "ADDICTED" if probValue > 0.5 else "NOT ADDICTED"

    if resultLabel == "ADDICTED":
        st.error(f"🚨 {resultLabel} (Risk: {probValue*100:.0f}%)")
    else:
        st.success(f"✅ {resultLabel} (Risk: {probValue*100:.0f}%)")

    st.session_state.attemptStore.append({
        "age": ageInput,
        "risk": probValue*100,
        "daily_hours": hoursUser,
        "status": resultLabel
    })

    m1,m2,m3 = st.columns(3)
    with m1: st.metric("Daily Usage",f"{hoursUser} hrs")
    with m2: st.metric("Checks/Hour",f"{checkUser}")
    with m3: st.metric("Age",f"{ageInput}y")

    radarData = pd.DataFrame({
        "Level":["Safe","Warning","Danger","YOUR SCORE"],
        "Risk":[30,60,100,probValue*100]
    })

    radarChart = px.line_polar(radarData,r="Risk",theta="Level",
                               color_discrete_sequence=["green","orange","red","purple"])

    st.plotly_chart(radarChart,use_container_width=True)

    csvData = pd.DataFrame(st.session_state.attemptStore).to_csv(index=False)
    st.download_button("📥 Download Results",csvData,"results.csv","text/csv")


if len(st.session_state.attemptStore) > 0:

    st.subheader("📈 Community Stats")

    totalCount = len(st.session_state.attemptStore)

    totalRiskSum = 0
    for entry in st.session_state.attemptStore:
        totalRiskSum += entry["risk"]

    avgRiskVal = totalRiskSum / totalCount

    col1,col2 = st.columns(2)
    with col1:
        st.metric("Total Tests",totalCount)
    with col2:
        st.metric("Avg Risk",f"{avgRiskVal:.0f}%")

st.markdown("---")
st.header("🆘 How to Reduce Phone Addiction")
st.info("Watch this recommended video for practical strategies.")

st.video("https://www.youtube.com/watch?v=ptrbG6675JA")

st.markdown("<div style='text-align:center;color:#666;'>Mobile Phone Addiction Predictor | BY : Kaushikun Krishna Kumar</div>", unsafe_allow_html=True)
'''

with open("app.py","w") as f:
    f.write(app_code)

!nohup streamlit run app.py --server.port 8501 --server.headless true >/dev/null 2>&1 &

import time
time.sleep(3)

public_url = ngrok.connect(8501)
print("🚀 LIVE APP LINK:", public_url)
