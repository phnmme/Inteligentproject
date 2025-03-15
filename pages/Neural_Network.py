import streamlit as st
from torchvision import datasets, models, transforms
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from PIL import Image

st.set_page_config(page_title="Cat vs Dog Classification", page_icon="🐱🐶", layout="centered")

st.title(" Cat vs Dog Classification")
st.markdown("**โปรแกรมนี้ใช้เทคนิคการจำแนกรูปภาพเพื่อแยกแยะว่าเป็นแมวหรือสุนัข**")

st.subheader(" การเตรียม Dataset")
st.write(
    "ข้อมูลที่ใช้มาจาก **Kaggle** โดยมีจำนวนรูปภาพทั้งหมด 4000 รูป (2000 รูปสำหรับแมวและ 2000 รูปสำหรับสุนัข)"
)
st.markdown("📌 **แหล่งที่มาของ Dataset:** [Kaggle - Cat and Dog Dataset](https://www.kaggle.com/tongpython/cat-and-dog)")
st.image("https://i.ytimg.com/vi/PV63uCaW8dc/maxresdefault.jpg", width=700)

st.subheader(" Convolutional Neural Network (CNN) คืออะไร?")
st.write(
    "CNN (Convolutional Neural Network) เป็นโครงข่ายประสาทเทียมเชิงลึก (Deep Learning) "
    "ที่ออกแบบมาเพื่อประมวลผลข้อมูลรูปภาพโดยเฉพาะ โดยใช้การคำนวณแบบ Convolution "
    "เพื่อทำการตรวจจับลักษณะเฉพาะของภาพ เช่น ขอบเขต และรูปแบบต่างๆ"
)

st.subheader(" Pre-trained ResNet คืออะไร?")
st.write(
    "Pre-trained ResNet (Residual Network) เป็นโมเดล CNN ที่ผ่านการฝึกบน Dataset ขนาดใหญ่ เช่น ImageNet "
    "ทำให้สามารถนำไปใช้งานได้ทันทีโดยไม่ต้องฝึกใหม่จากศูนย์ (Training from Scratch)"
    " ResNet ใช้ Residual Blocks ที่ช่วยลดปัญหา Gradient Vanishing และช่วยให้การฝึกโมเดลลึกๆ เป็นไปได้ง่ายขึ้น"
)

st.subheader(" ขั้นตอนในการพัฒนาโมเดล")

st.divider()

st.markdown("### 🔹 **ขั้นตอนที่ 1: โหลด Dataset และแปลงรูปภาพเป็น Tensor**")
code1 = """
train_dir = 'Path ของ Dataset ที่เตรียมไว้'
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_data = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
"""
st.code(code1, language='python')

st.markdown("### 🔹 **ขั้นตอนที่ 2: สร้างโมเดล CNN ด้วยฟังก์ชัน create_model()**")
code2 = """
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to('cpu')
"""
st.code(code2, language='python')

st.markdown("### 🔹 **ขั้นตอนที่ 3: ฝึกโมเดลด้วยฟังก์ชัน train_model()**")
code3 = """
model = create_model()
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001) 

progress_bar = st.progress(0) 

for epoch in range(epochs): 
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader): 
        inputs, labels = inputs.to('cpu'), labels.to('cpu')  

        optimizer.zero_grad()  
        outputs = model(inputs) 
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()  
        optimizer.step() 
        running_loss += loss.item()

        progress = ((epoch * len(train_loader)) + (i + 1)) / (epochs * len(train_loader))
        progress_bar.progress(int(progress * 100))  

    st.write(f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss / len(train_loader):.4f}")

save_model(model)
return model
"""
st.code(code3, language='python')

st.markdown("### 🔹 **ขั้นตอนที่ 4: ทดสอบโมเดลด้วยรูปภาพที่ไม่เคยเห็นมาก่อน**")
st.write("ใช้ฟังก์ชัน `predict()` เพื่อทำนายผลว่าเป็นแมวหรือสุนัข")
code4 = """
def predict(model, uploaded_image):
    img = Image.open(uploaded_image).resize((150, 150))
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(img)
        prediction = torch.sigmoid(output).item()
    label = "Cat" if prediction >= 0.5 else "Dog"
    confidence = prediction if label == "Cat" else 1 - prediction
    return f"{label} ({confidence:.2%} confidence)"
"""
st.code(code4, language='python')

st.divider()
