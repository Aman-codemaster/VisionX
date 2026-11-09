def classify_image(image, model):

    # Get predictions from the model
    results = model(image, top_k=None)

    # Sort predictions by confidence
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # Get the highest prediction
    best_result = results[0]

    label = best_result["label"]
    confidence = best_result["score"]

    return label, confidence, results