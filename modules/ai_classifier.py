def classify_image(image, detector):

    # Run Smaug on the uploaded image
    score = detector(image.convert("RGB"))

    # Convert score into a percentage
    ai_probability = float(score) * 100

    # Classify the image
    if score >= 0.5:
        prediction = "AI-Generated"
    else:
        prediction = "Likely Real"

    return prediction, ai_probability
