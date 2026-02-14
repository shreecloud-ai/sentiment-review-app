from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model.predict import predict_sentiment   # ← remove predict_batch

app = FastAPI(
    title="Product Review Sentiment Analyzer",
    description="Simple NLP API to classify review sentiment (positive / neutral / negative)",
    version="0.1.0"
)


class ReviewRequest(BaseModel):
    text: str


class BatchReviewRequest(BaseModel):
    texts: list[str]


@app.get("/")
def root():
    return {"message": "Sentiment Analyzer API is running. Try /docs for Swagger UI"}


@app.post("/predict", response_model=dict)
def predict_single(review: ReviewRequest):
    try:
        result = predict_sentiment(review.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/predict_batch", response_model=list[dict])
# def predict_multiple(reviews: BatchReviewRequest):
#     try:
#         results = predict_batch(reviews.texts)
#         return results
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# Optional health check
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}