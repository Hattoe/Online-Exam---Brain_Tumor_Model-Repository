import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import load_model, NUM_CLASSES

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st

from src.model import load_model, NUM_CLASSES


# ---------- Paths & constants ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "models" / "best_model.pth"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE = "cpu"  # Streamlit Cloud won't give you GPU


# ---------- Preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@st.cache_resource
def get_model():
    model = load_model(str(WEIGHTS_PATH), device=DEVICE)
    return model


def preprocess_image(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)  # add batch dim
    return tensor


def predict(img: Image.Image):
    model = get_model()
    x = preprocess_image(img).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = probs.argmax()
    return CLASS_NAMES[pred_idx], probs


# ---------- Streamlit UI ----------
st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI image, I’ll classify it into one of four classes:")

st.write(", ".join(CLASS_NAMES))

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        label, probs = predict(image)
        st.subheader(f"Prediction: {label}")

        st.write("Probabilities:")
        for cls, p in zip(CLASS_NAMES, probs):
            st.write(f"- {cls}: {p:.3f}")
