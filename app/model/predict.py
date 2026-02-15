import joblib
import numpy as np

# Load model once (global)
MODEL_PATH = "models/sentiment_model.joblib"

try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"[STARTUP] Model loaded successfully from {MODEL_PATH} (size: {os.path.getsize(MODEL_PATH)/1e6:.1f} MB)")
except Exception as e:
    print(f"[STARTUP ERROR] Failed to load model: {str(e)}")
    pipeline = None


def predict_sentiment(text: str):
    """
    Predict sentiment + top words explanation.
    Returns dict safe for API.
    """
    if pipeline is None:
        return {"error": "Model not loaded"}

    if not text or len(text.strip()) < 3:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            "top_positive_words": [],
            "message": "Review too short"
        }

    try:
        # Core prediction
        pred_array = pipeline.predict([text])
        pred = pred_array[0]  # this is the line that must exist before using 'pred'

        probs_array = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_.tolist()
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, probs_array)}
        confidence = max(probabilities.values())

        # Simple top words explanation (using coefficients)
        top_words = []
        message = "No explanation available"

        try:
            tfidf = pipeline.named_steps['tfidf']
            features = tfidf.get_feature_names_out()
            coefs = pipeline.named_steps['clf'].coef_

            # Find index of predicted class
            pred_idx = np.where(classes == pred)[0][0]
            class_coefs = coefs[pred_idx]

            # Top positive coefficients (words pushing toward this class)
            top_indices = class_coefs.argsort()[-10:][::-1]  # top 10
            top_words = [
                features[i] for i in top_indices
                if class_coefs[i] > 0.05  # threshold to filter weak/noisy terms
            ][:5]

            if top_words:
                message = f"Key words driving the '{pred}' prediction: {', '.join(top_words)}"
        except Exception as expl_err:
            print(f"Explanation error: {expl_err}")
            # Continue without explanation

        return {
            "sentiment": pred,
            "confidence": confidence,
            "probabilities": probabilities,
            "top_positive_words": top_words,
            "message": message
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}
def predict_batch(texts: list[str]):
    """
    Predict sentiment for multiple reviews at once.
    Returns list of dicts.
    """
    if pipeline is None:
        return [{"error": "Model not loaded"}] * len(texts)

    if not texts:
        return []

    try:
        preds = pipeline.predict(texts)
        probs_list = pipeline.predict_proba(texts)

        results = []
        classes = pipeline.classes_.tolist()

        for pred, probs in zip(preds, probs_list):
            probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
            confidence = max(probabilities.values())

            # Optional: add top words per review (can be slow for large batches)
            top_words = []  # skip for batch or implement if needed

            results.append({
                "sentiment": pred,
                "confidence": confidence,
                "probabilities": probabilities,
                "top_positive_words": top_words,
                "message": ""  # or add simple message
            })

        return results

    except Exception as e:
        print(f"Batch prediction error: {e}")
        return [{"error": str(e)}] * len(texts)