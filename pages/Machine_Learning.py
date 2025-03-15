import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="HDPredictor", page_icon="❤️", layout="centered")

st.title("HDPredictor - Heart Disease Prediction")
st.markdown("**โปรแกรมนี้สามารถทำนายว่าคุณมีโอกาสเป็นโรคหัวใจหรือไม่ จากข้อมูลสุขภาพของคุณ**")

st.subheader(" การเตรียม Dataset")
st.write(
    "ข้อมูลที่ใช้มาจาก **Kaggle** โดยเป็น Dataset ที่เกี่ยวกับโรคหัวใจ "
    "ซึ่งรวบรวมปัจจัยสุขภาพที่อาจมีผลต่อโรคหัวใจ เช่น อายุ, คอเลสเตอรอล, ความดันโลหิต เป็นต้น"
)
st.markdown("📌 **แหล่งที่มาของ Dataset:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)")
st.image("https://hdmall.co.th/blog/wp-content/uploads/2024/09/cardiovascular-treatment-comparison-01-scaled.jpg.webp", width=700)

st.subheader(" K-Nearest Neighbors (KNN) คืออะไร?")
st.write(
    "KNN (K-Nearest Neighbors) เป็นอัลกอริธึมการเรียนรู้ของเครื่อง (Machine Learning) "
    "ประเภท Supervised Learning ที่ใช้สำหรับการ **จำแนกประเภท (Classification)** และ **การถดถอย (Regression)** "
    "แต่โดยทั่วไปนิยมใช้สำหรับ **การจำแนกประเภท** มากกว่า"
)
st.write(
    "หลักการทำงานของ KNN คือการหาจุดข้อมูลที่ใกล้เคียง (Neighbor) จำนวน **K จุด** "
    "แล้วใช้ค่าที่พบมากที่สุดในกลุ่มนั้นมาทำนายผลลัพธ์ของข้อมูลใหม่"
)

st.subheader(" ขั้นตอนในการพัฒนาโมเดล")

st.divider()

st.markdown("### 🔹 **ขั้นตอนที่ 1: โหลด Dataset และแสดงตัวอย่างข้อมูล**")
code1 = """
df = pd.read_csv(DATA_PATH)
st.dataframe(df.head(10))
"""
st.code(code1, language='python')

st.markdown("### 🔹 **ขั้นตอนที่ 2: แยก Features และ Target จาก Dataset**")
code2 = """
X = df.drop(columns=["target"])
y = df["target"]
"""
st.code(code2, language='python')

st.markdown("### 🔹 **ขั้นตอนที่ 3: แบ่งข้อมูลเป็น Train/Test**")
code3 = """
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
st.code(code3, language='python')

st.markdown("### 🔹 **ขั้นตอนที่ 4: ทำการ Standardization ของข้อมูล**")
code4 = """
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""
st.code(code4, language='python')

st.markdown("### 🔹 **ขั้นตอนที่ 5: Train โมเดล KNN (k=5)**")
code5 = """
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train_scaled, y_train)       
"""
st.code(code5, language='python')

st.markdown("### 🔹 **ขั้นตอนที่ 6: รับข้อมูลจากผู้ใช้ และทำนายผล**")
code6 = """
user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
user_input_scaled = scaler.transform(user_input)

prediction = knn.predict(user_input_scaled)
probability = knn.predict_proba(user_input_scaled)[0]
"""
st.code(code6, language='python')

st.divider()
