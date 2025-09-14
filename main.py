import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import streamlit as st

# --- Классы ---
FINAL_CLASSES = [
    "dent",
    "scratch",
    "broken glass",
    "lost parts",
    "punctured",
    "torn",
    "broken lights",
    "non-damage",
]

MODEL_NAME = "resnet50"
CHECKPOINT = "vehide_cls8_resnet50_best.pt"
IMG_SIZE = 448
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Модель ---
class VehicleDamageModel(nn.Module):
    def __init__(self, num_classes=3, model_name="resnet50"):
        super().__init__()
        if model_name == "resnet50":
            self.backbone = models.resnet50(weights=None)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=None)
            self.backbone.classifier[1] = nn.Linear(
                self.backbone.classifier[1].in_features, num_classes
            )
        else:
            raise ValueError(model_name)

    def forward(self, x):
        return self.backbone(x)


# --- Загружаем модель ---
@st.cache_resource
def load_model():
    model = VehicleDamageModel(num_classes=len(FINAL_CLASSES), model_name=MODEL_NAME).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        has_backbone = any(k.startswith("backbone.") for k in state.keys())
        if not has_backbone:
            state = {f"backbone.{k}": v for k, v in state.items()}
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            state2 = {k.replace("backbone.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state2, strict=False)

    model.eval()
    return model


# --- Трансформации ---
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --- Интерфейс Streamlit ---
st.title("🚗 Vehicle Damage Classifier")
st.write("Загрузите фото автомобиля, чтобы определить тип повреждения.")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # показываем картинку
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Загруженное изображение", use_column_width=True)

    # инференс
    model = load_model()
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        pred_label = FINAL_CLASSES[pred_idx]
        pred_conf = float(probs[pred_idx])

    st.subheader(f"🔮 Предсказание: **{pred_label}** ({pred_conf:.2%})")

    st.write("### Вероятности по классам:")
    for cls, p in zip(FINAL_CLASSES, probs.tolist()):
        st.write(f"- {cls}: {p:.2%}")
