import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

DATA_PATH = "dataset/heart.csv"

df = pd.read_csv(DATA_PATH)

st.title("Heart Disease Prediction")
st.write("**ตัวอย่างข้อมูลใน Dataset**")
st.dataframe(df.head(10))

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

st.write("กรอกข้อมูลสุขภาพของคุณเพื่อประเมินความเสี่ยง")

age = st.number_input("อายุ (Age)", min_value=20, max_value=100, value=50)
sex = st.selectbox("เพศ (Sex)", [0, 1], format_func=lambda x: "ชาย" if x == 1 else "หญิง")
cp = st.selectbox("อาการเจ็บหน้าอก (Chest Pain Type)", [0, 1, 2, 3])
trestbps = st.number_input("ความดันโลหิตขณะพัก (Resting Blood Pressure)", min_value=80, max_value=200, value=120)
chol = st.number_input("ระดับคอเลสเตอรอล (Cholesterol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("น้ำตาลในเลือดสูงกว่า 120 mg/dl (Fasting Blood Sugar)", [0, 1], format_func=lambda x: "สูง" if x == 1 else "ปกติ")
restecg = st.selectbox("ผล ECG (Resting ECG)", [0, 1, 2])
thalach = st.number_input("อัตราการเต้นของหัวใจสูงสุด (Max Heart Rate)", min_value=60, max_value=220, value=150)
exang = st.selectbox("มีอาการเจ็บหน้าอกขณะออกกำลังกาย (Exercise Angina)", [0, 1], format_func=lambda x: "มี" if x == 1 else "ไม่มี")
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope ของ ST Segment", [0, 1, 2])
ca = st.selectbox("จำนวนหลอดเลือดที่ถูกบล็อก", [0, 1, 2, 3, 4])
thal = st.selectbox("ผล Thalassemia", [0, 1, 2, 3])

user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
user_input_scaled = scaler.transform(user_input)

if st.button("ทำนายผล"):
    prediction = knn.predict(user_input_scaled)
    probability = knn.predict_proba(user_input_scaled)[0]

    if prediction[0] == 1:
        st.error(f"มีความเสี่ยงโรคหัวใจ ({probability[1]*100:.2f}%)")
    else:
        st.success(f"ปกติ ({probability[0]*100:.2f}%)")
