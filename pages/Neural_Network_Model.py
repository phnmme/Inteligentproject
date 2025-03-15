import streamlit as st
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import os

MODEL_PATH = "dataset/cat_vs_dog.pth"

def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to('cpu')

def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)
    st.success("โหลด model สำเร็จ!")

def load_model():
    model = create_model()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        st.success("Model loaded successfully!")
    return model

# def train_model(epochs):
#     train_dir = 'D:/KMUTNB-CS/Inteligentproject/dataset/training_set'
#     transform = transforms.Compose([
#         transforms.Resize((150, 150)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     train_data = datasets.ImageFolder(train_dir, transform=transform)
#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

#     model = create_model()
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     progress_bar = st.progress(0)

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for i, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to('cpu'), labels.to('cpu')
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs.squeeze(), labels.float())
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             progress = ((epoch * len(train_loader)) + (i + 1)) / (epochs * len(train_loader))
#             progress_bar.progress(int(progress * 100))
#         st.write(f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss / len(train_loader):.4f}")
    
#     save_model(model)
#     return model

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
    label = "Dog" if prediction >= 0.5 else "Cat"
    confidence = prediction if label == "Dog" else 1 - prediction
    return f"{label} ({confidence:.2%} confidence)"

st.title('Cat vs Dog Classifier')
st.write('อัปโหลดรูปภาพของแมวหรือสุนัขเพื่อทำนายผล')

uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"])
# epochs = st.slider("เลือกจำนวน Epochs", min_value=1, max_value=50, value=10, step=1)

# if st.button('เริ่มการฝึกโมเดล'):
#     st.write(f"เริ่มการฝึกโมเดล {epochs} รอบ Epochs")
#     train_model(epochs)
#     st.success("Training completed!")

model = load_model()

if uploaded_file is not None:
    if st.button('แยกแมวและสุนัข'):
        result = predict(model, uploaded_file)
        st.write(f'ผลลัพธ์: {result}')
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
