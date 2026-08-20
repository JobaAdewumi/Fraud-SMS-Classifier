import os
from fastapi.responses import FileResponse
import joblib
import re
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
import uvicorn

from utils import SMSTextCleaner

# Initialize Application
app = FastAPI(title="Nigerian SMS Fraud Classifier API")

# Load the Trained Model
try:
    model_pipeline = joblib.load('sms_fraud_model.joblib')
    vectorizer = model_pipeline.named_steps['tfidf']
    classifier = model_pipeline.named_steps['clf']
except Exception as e:
    raise RuntimeError("Model file not found. Run the notebook first.") from e

# Define Data Schemas
class MessageRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    fraud_probability: float
    safe_probability: float
    keyword_insights: dict

# Define the ML Endpoint
@app.post("/api/predict", response_model=PredictionResponse)
async def predict_sms(request: MessageRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Message text cannot be empty.")

    text = request.text
    
    # Generate Probabilities (The pipeline automatically handles the regex cleaning here)
    probabilities = model_pipeline.predict_proba([text])[0]
    safe_prob, fraud_prob = probabilities[0], probabilities[1]
    
    is_fraud = fraud_prob > 0.5
    label = "Fraudulent" if is_fraud else "Safe"
    confidence = max(safe_prob, fraud_prob)

    # Local Explainability (Updated for the new pipeline structure)
    # Use pipeline[:-1] to pass the text through the cleaner AND the vectorizer
    transformed_text = model_pipeline[:-1].transform([text])
    feature_indices = transformed_text.nonzero()[1]
    feature_names = vectorizer.get_feature_names_out()
    
    word_contributions = {}
    for idx in feature_indices:
        word = feature_names[idx]
        weight = classifier.coef_[0][idx]
        word_contributions[word] = float(weight)

    # Sort keywords by their fraud impact weight
    sorted_insights = dict(sorted(word_contributions.items(), key=lambda item: item[1], reverse=True))

    return PredictionResponse(
        label=label,
        confidence=round(confidence * 100, 2),
        fraud_probability=round(fraud_prob * 100, 2),
        safe_probability=round(safe_prob * 100, 2),
        keyword_insights=sorted_insights
    )

@app.get("/logo-white.png")
async def get_logo():
    logo_path = os.path.join(os.path.dirname(__file__), "logo-white.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path)
    raise HTTPException(status_code=404, detail="Logo not found")

# Mount Frontend
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Build absolute paths for Vercel's environment
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# STATIC_DIR = os.path.join(BASE_DIR, "static")

# @app.get("/")
# async def serve_frontend():
#     """Serves the main HTML interface."""
#     index_path = os.path.join(STATIC_DIR, "index.html")
#     if not os.path.exists(index_path):
#         raise HTTPException(status_code=404, detail="Frontend not found")
#     return FileResponse(index_path)

# @app.get("/{filename}")
# async def serve_static_assets(filename: str):
#     """Serves static assets."""
#     file_path = os.path.join(STATIC_DIR, filename)
#     if os.path.exists(file_path):
#         return FileResponse(file_path)
#     raise HTTPException(status_code=404, detail="File not found")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
