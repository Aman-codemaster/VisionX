import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
# Model configuration
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "deepfake"
    / "pytorch_model.bin"
)
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(
    (260, 260),
    interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
def load_deepfake_model():

    model = models.efficientnet_b2(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1408, 2)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model


def predict_deepfake(image, face_model, deepfake_model):

    # Detect faces using our existing YOLO face model
    results = face_model.predict(
        image,
        conf=0.25,
        verbose=False
    )

    boxes = results[0].boxes

    if len(boxes) == 0:
        return {"error": "No face detected."}

    # Select the largest detected face
    areas = boxes.xyxy[:, 2] - boxes.xyxy[:, 0]
    heights = boxes.xyxy[:, 3] - boxes.xyxy[:, 1]
    largest_index = torch.argmax(areas * heights).item()

    x1, y1, x2, y2 = boxes.xyxy[
        largest_index
    ].cpu().tolist()

    # Crop the face
    # Add 20-pixel padding around the face
    padding = 20

    image_width, image_height = image.size

    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(image_width, int(x2) + padding)
    y2 = min(image_height, int(y2) + padding)

    face = image.crop((x1, y1, x2, y2))

    # Preprocess the face
    face_tensor = transform(face).unsqueeze(0).to(DEVICE)

    # Run deepfake classification
    with torch.no_grad():
        output = deepfake_model(face_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    fake_probability = probabilities[0].item() * 100
    real_probability = probabilities[1].item() * 100

    prediction = (
        "FAKE" if fake_probability > real_probability else "REAL"
    )

    return {
        "prediction": prediction,
        "fake_confidence": round(fake_probability, 2),
        "real_confidence": round(real_probability, 2)
    }
