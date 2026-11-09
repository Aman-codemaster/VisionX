import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from retinaface import RetinaFace
from pathlib import Path
from typing import Optional

# --- Configuration ---
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "pytorch_model.bin"
)  # Updated to match your new HF filename
_IN_FEATURES = 1408
_DROPOUT = 0.3
_NUM_CLASSES = 2
_INPUT_SIZE = 260
_CONFIDENCE_THRESHOLD = 0.90
_MIN_FACE_PX = 50
_PADDING = 20

# --- Transform ---
_transform = transforms.Compose([
    transforms.Resize((_INPUT_SIZE, _INPUT_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 1. Load Architecture ---
def load_model() -> tuple[nn.Module, torch.device]:
    # Universal Hardware Routing
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    net = models.efficientnet_b2(weights=None)
    net.classifier = nn.Sequential(
        nn.Dropout(_DROPOUT),
        nn.Linear(_IN_FEATURES, _NUM_CLASSES),
    )
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    
    net.to(device)
    net.eval()
    
    return net, device

MODEL, DEVICE = load_model()

# --- 2. Extract & Preprocess ---
def detect_and_crop_face(image_path: str) -> Optional[torch.Tensor]:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not load image at {image_path}")

    detections = RetinaFace.detect_faces(image_bgr)
    if not isinstance(detections, dict):
        return None

    best_conf = -1.0
    best_box = None

    for face_data in detections.values():
        conf = float(face_data.get("score", 0.0))
        if conf < _CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = face_data["facial_area"]
        w, h = x2 - x1, y2 - y1
        if w < _MIN_FACE_PX or h < _MIN_FACE_PX:
            continue

        if conf > best_conf:
            best_conf = conf
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = best_box
    x1 = max(0, x1 - _PADDING)
    y1 = max(0, y1 - _PADDING)
    x2 = min(W, x2 + _PADDING)
    y2 = min(H, y2 + _PADDING)

    crop_bgr = image_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_face = Image.fromarray(crop_rgb)
    
    return _transform(pil_face).unsqueeze(0)

# --- 3. Execute Prediction ---
def predict_deepfake(image_path: str) -> dict:
    face_tensor = detect_and_crop_face(image_path)
    
    if face_tensor is None:
        return {"error": "No face detected in the image meeting confidence thresholds."}
        
    face_tensor = face_tensor.to(DEVICE)
    
    with torch.no_grad():
        logits = MODEL(face_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        
    fake_prob = probs[0].item() * 100
    real_prob = probs[1].item() * 100
    predicted_idx = int(torch.argmax(probs).item())
    
    prediction = "REAL" if predicted_idx == 1 else "FAKE"
    
    return {
        "prediction": prediction,
        "fake_confidence": f"{fake_prob:.2f}%",
        "real_confidence": f"{real_prob:.2f}%"
    }