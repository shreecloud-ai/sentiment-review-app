from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model.predict import predict_sentiment

app = FastAPI(title="Sentiment Prediction API")

class ReviewRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(review: ReviewRequest):
    try:
        result = predict_sentiment(review.text)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))