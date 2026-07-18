import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="wide")

model=joblib.load("gbr_model.pkl")
scaler=joblib.load("scaler.pkl")

st.markdown(r"""
<style>
#MainMenu,header,footer{visibility:hidden;}
.stApp{background:linear-gradient(135deg,#0b1026,#1d4ed8,#3b82f6);}
.hero{padding:28px;border-radius:22px;background:rgba(255,255,255,.08);backdrop-filter:blur(12px);}
.hero h1{text-align:center;color:white;}
.hero p{text-align:center;color:#dbeafe;}
.stButton>button{width:100%;height:56px;border-radius:14px;font-size:20px;font-weight:bold;background:#2563eb;color:white;}
.result{padding:20px;border-radius:16px;background:#16a34a;color:white;text-align:center;font-size:28px;font-weight:bold;}
</style>
""",unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>💰 Employee Salary Predictor</h1><p>AI Powered Salary Prediction</p></div>',unsafe_allow_html=True)

c1,c2=st.columns(2)
with c1:
    age=st.number_input("🎂 Age",18,70,25)
    gender=st.selectbox("👤 Gender",["Female","Male"])
    education=st.selectbox("🎓 Education",["Diploma","Bachelor","Master","PhD"])
    exp=st.number_input("💼 Experience (Years)",0.0,50.0,2.0)
    dept=st.selectbox("🏢 Department",["Operations","IT","Sales","HR","Marketing"])
    level=st.selectbox("📈 Job Level",["Junior","Mid","Senior","Lead","Manager"])
    perf=st.slider("⭐ Performance Rating",1,5,3,1)
with c2:
    cert=st.number_input("📜 Certifications",0,20,0)
    over=st.number_input("⏰ Overtime Hours",0,200,10)
    remote=st.selectbox("🏠 Remote Work",["No","Yes"])
    city=st.selectbox("🌍 City",["Hyderabad","Mumbai","Chennai","Delhi"])
    tenure=st.number_input("📅 Company Tenure",0.0,50.0,2.0)
    proj=st.number_input("📂 Projects Completed",0,100,5)
    skill=st.slider("🧠 Skill Score",0,100,50,1)

gender={"Female":0,"Male":1}[gender]
education={"Diploma":0,"Bachelor":1,"PhD":2,"Master":3}[education]
dept={"Operations":0,"IT":1,"Sales":2,"HR":3,"Marketing":4}[dept]
level={"Junior":0,"Manager":1,"Mid":2,"Senior":3,"Lead":4}[level]
remote={"No":0,"Yes":1}[remote]
city={"Hyderabad":0,"Mumbai":1,"Chennai":2,"Delhi":3}[city]

df=pd.DataFrame({"Age":[age],"Gender":[gender],"Education":[education],"Experience_Years":[exp],"Department":[dept],"Job_Level":[level],"Performance_Rating":[perf],"Certifications":[cert],"Overtime_Hours":[over],"Remote_Work":[remote],"City":[city],"Company_Tenure":[tenure],"Projects_Completed":[proj],"Skill_Score":[skill]})
cols=["Experience_Years","Performance_Rating","Certifications","Overtime_Hours","Company_Tenure","Projects_Completed","Skill_Score"]
df[cols]=scaler.transform(df[cols])

if st.button("🚀 Predict Salary"):
    pred=model.predict(df)[0]
    st.markdown(f'<div class="result">💰 Estimated Annual Salary<br><br>₹ {pred:.2f} LPA</div>',unsafe_allow_html=True)

st.caption("Developed by Naved Alam")
