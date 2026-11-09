import cv2
from PIL import Image
from collections import Counter


def detect_objects(image, model, confidence=0.25):

    # Run YOLO detection
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # Get annotated image
    annotated_image = result.plot()

    # Convert OpenCV BGR image to RGB
    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    # Convert to PIL image
    annotated_image = Image.fromarray(annotated_image)

    # Extract detected objects
    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence_score = float(box.conf[0])
        object_name = result.names[class_id]

        detections.append({
            "object": object_name,
            "confidence": round(confidence_score * 100, 2)
        })

    # Count objects by class
    object_counts = Counter(
        item["object"] for item in detections
    )

    return annotated_image, detections, object_counts