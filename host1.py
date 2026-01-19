import os
os.system("git lfs pull")

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from groq import Groq
import cv2




device = "cuda" if torch.cuda.is_available() else "cpu"

def load_classes(path="class_names.txt"):
    with open(path, "r") as f:
        return [line.strip() for line in f]

classes = load_classes()
num_classes = len(classes)

## CUSTOM CNN MODEL
class SimpleCNN(nn.Module):
    def __init__(self, num_classes = 53):
        super().__init__()
        self.features = nn.Sequential(
            # Block-1: B, 3, 256, 256 → B, 32, 128, 128
            nn.Conv2d(in_channels = 3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block-2: B, 32, 128, 128 → B, 64, 64, 64
            nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block-3: B, 64, 64, 64 → B, 128, 32, 32
            nn.Conv2d(in_channels = 64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block-4: B, 128, 32, 32 → B, 256, 16, 16
            nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block-5: B, 256, 16, 16 → B, 512, 8, 8
            nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(), # B, 512
            nn.Dropout(p=0.4),
            nn.Linear(in_features=512, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


## LOAD
@st.cache_resource
def load_model(model_choice):
    if model_choice == "ResNet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        checkpoint = torch.load("best_model_resnet34.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state"])

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    else:  
        model = SimpleCNN(num_classes)

        checkpoint = torch.load("best_model_custom.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state"])

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    model.to(device)
    model.eval()
    return model, transform

# LLM
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def ask_llm(pred_batu):
    prompt = f"""
    Jelaskan ciri fisik, sifat fisik, proses keterbentukan, dan mineral penyusun
    dari batuan {pred_batu}. Gunakan bahasa Indonesia secara akademis Geologi.
    Fokuskan proses keterbentukan. Maksimal 200 kata.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15
    )
    return response.choices[0].message.content

# SIDEBAR
st.sidebar.title("Setting")

model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["ResNet34", "Custom CNN"]
)

menu = st.sidebar.radio(
    "Input Mode :",
    ["Upload", "Webcam"]
)

model, transform = load_model(model_choice)

# UI
st.title("Rock Classification")
st.caption(f"Active Model : **{model_choice}**")


if menu == "Upload":
    file = st.file_uploader("Upload Rock Image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=300, caption="Input")

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_k = 3
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        pred_idx = top_indices[0]
        pred_batu = classes[pred_idx]
        confidence = probabilities[pred_idx]

        st.success(f"Predict: **{pred_batu}** ({confidence*100:.2f}%)")
      
        st.subheader("📉 Confidence")

        for i in top_indices:
            st.write(f"**{classes[i]}** — {probabilities[i]*100:.2f}%")
            st.progress(float(probabilities[i]))

        st.subheader("📚 Explanation")
        with st.spinner("Loading..."):
            st.write(ask_llm(pred_batu))

else:

    st.subheader("📷 Take a Photo")

    photo = st.camera_input("Click to take photo")

    if photo is not None:
        img = Image.open(photo).convert("RGB")
        st.image(img, width=300, caption="Foto Webcam")

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_k = 3
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        pred_idx = top_indices[0]
        pred_batu = classes[pred_idx]
        confidence = probabilities[pred_idx]

        st.success(f"Predict: **{pred_batu}** ({confidence*100:.2f}%)")

        st.subheader("📉 Confidence")
        for i in top_indices:
            st.write(f"**{classes[i]}** — {probabilities[i]*100:.2f}%")
            st.progress(float(probabilities[i]))

        st.subheader("📚 Explanation")
        with st.spinner("Loading..."):
            st.write(ask_llm(pred_batu))

