
import cv2
from PIL import Image


def detect_faces(image, model, confidence=0.25):

    # Run the face detection model
    results = model.predict(
        source=image,
        imgsz=1280,
        conf=confidence,
        max_det=300,
        verbose=False
    )

    # Get detection result
    result = results[0]

    # Draw bounding boxes
    annotated_image = result.plot()

    # Convert BGR image to RGB
    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    # Count detected faces
    face_count = len(result.boxes)

    # Store face information
    face_details = []

    for box in result.boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        score = float(box.conf[0].cpu())

        face_details.append({
            "X": int(x1),
            "Y": int(y1),
            "Width": int(x2 - x1),
            "Height": int(y2 - y1),
            "Confidence": round(score, 3)
        })

    return (
        Image.fromarray(annotated_image),
        face_count,
        face_details
    )
