
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from io import BytesIO

from modules.object_detection import detect_objects
from modules.face_detection import detect_faces
from transformers import pipeline
from modules.ai_classifier import classify_image

# =====================================
# PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

OBJECT_MODEL_PATH = BASE_DIR / "yolo11n.pt"

FACE_MODEL_PATH = (
    BASE_DIR / "models" / "yolov8n-face-lindevs.pt"
)


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="VisionX AI",
    page_icon="👁️",
    layout="wide"
)


# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_yolo_model():
    return YOLO(str(OBJECT_MODEL_PATH))


@st.cache_resource
def load_face_model():
    return YOLO(str(FACE_MODEL_PATH))

@st.cache_resource
def load_ai_model():
    model = pipeline(
        "image-classification",
        model="aaronkantrowitz/ai-human-generated-image-detection"
    )
    return model

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("👁️ VisionX AI")
st.sidebar.write("AI-Powered Image Analysis")

selected_tool = st.sidebar.radio(
    "Select Analysis Tool",
    [
        "Object Detection",
        "Face Detection",
        "Real vs AI Image"
    ]
)


# =====================================
# MAIN INTERFACE
# =====================================

st.title("👁️ VisionX AI")
st.subheader("Intelligent Image Analysis Platform")

st.write(
    "Upload an image and analyse it using "
    "computer vision and artificial intelligence."
)

st.divider()
st.header(selected_tool)


# =====================================
# IMAGE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    # Display a smaller centered preview
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(image, width=350)

    width, height = image.size

    with st.expander("Image Information"):

        info1, info2, info3 = st.columns(3)

        info1.metric("Width", f"{width} px")
        info2.metric("Height", f"{height} px")
        info3.metric("Format", uploaded_file.type or "Unknown")


    # =================================
    # OBJECT DETECTION
    # =================================

    if selected_tool == "Object Detection":

        confidence = st.slider(
            "Detection Confidence",
            0.10,
            0.90,
            0.25,
            0.05
        )

        if st.button("Analyse Image"):

            try:
                with st.spinner("Detecting objects..."):

                    model = load_yolo_model()

                    annotated, detections, counts = detect_objects(
                        image,
                        model,
                        confidence
                    )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)

                with col2:
                    st.subheader("Detection Results")
                    st.image(annotated, use_container_width=True)

                st.divider()

                st.metric(
                    "Total Objects Detected",
                    len(detections)
                )

                if counts:

                    st.subheader("Object Counts")

                    for name, count in counts.items():
                        st.write(f"**{name}:** {count}")

                    st.subheader("Detection Details")

                    st.dataframe(
                        detections,
                        use_container_width=True
                    )

                else:
                    st.warning("No objects detected.")

                # Download result
                buffer = BytesIO()
                annotated.save(buffer, format="PNG")

                st.download_button(
                    "Download Annotated Image",
                    data=buffer.getvalue(),
                    file_name="visionx_objects.png",
                    mime="image/png"
                )

            except Exception as error:
                st.error(f"Object detection failed: {error}")


    # =================================
    # FACE DETECTION
    # =================================

    elif selected_tool == "Face Detection":

        confidence = st.slider(
            "Face Detection Confidence",
            0.05,
            0.90,
            0.25,
            0.05
        )

        if st.button("Detect Faces"):

            try:
                with st.spinner("Detecting faces..."):

                    if not FACE_MODEL_PATH.exists():
                        raise FileNotFoundError(
                            "Face model not found in the models folder."
                        )

                    face_model = load_face_model()

                    annotated, face_count, face_details = detect_faces(
                        image,
                        face_model,
                        confidence
                    )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)

                with col2:
                    st.subheader("Face Detection Results")
                    st.image(annotated, use_container_width=True)

                st.divider()

                st.metric("Total Faces Detected", face_count)

                if face_details:

                    st.subheader("Face Details")

                    st.dataframe(
                        face_details,
                        use_container_width=True
                    )

                else:
                    st.warning("No faces detected.")

                # Download result
                buffer = BytesIO()
                annotated.save(buffer, format="PNG")

                st.download_button(
                    "Download Face Detection Image",
                    data=buffer.getvalue(),
                    file_name="visionx_faces.png",
                    mime="image/png"
                )

            except Exception as error:
                st.error(f"Face detection failed: {error}")


    # =================================
    # REAL VS AI IMAGE
    # =================================

    elif selected_tool == "Real vs AI Image":


        if st.button("Analyse Image", key="ai_analysis"):

            try:
                with st.spinner("Analysing image..."):

                    ai_model = load_ai_model()

                    label, confidence, results = classify_image(
                        image,
                        ai_model
                    )

                st.subheader("Classification Result")

                st.write(f"Prediction: **{label}**")

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.subheader("All Predictions")

                for result in results:
                    st.write(
                        f"{result['label']}: "
                        f"{result['score'] * 100:.2f}%"
                    )

            except Exception as error:
                st.error(f"Classification failed: {error}")

else:

    st.info("Upload an image to begin analysis.")
