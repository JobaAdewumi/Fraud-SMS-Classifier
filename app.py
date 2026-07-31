import joblib
import re
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
import uvicorn

# Define the Custom Cleaner (MUST match the notebook exactly)
class SMSTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.patterns = [
            (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' __URL__ '),
            (r'(?:\+?234|0)[789][01]\d{8}', ' __PHONE__ '),
            (r'\b\d{10}\b', ' __ACCOUNT_NUM__ '),
            (r'(?:N|₦)?\d+(?:,\d+)*(?:\.\d+)?(?:k|m|b|\s*naira)?', ' __MONEY__ '),
            (r'\*\d+(?:\*\d+)*#', ' __USSD__ ')
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cleaned_X = []
        for text in X:
            text = str(text).lower()
            for pattern, replacement in self.patterns:
                text = re.sub(pattern, replacement, text)
            text = re.sub(r'\s+', ' ', text).strip()
            cleaned_X.append(text)
        return cleaned_X

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

# Mount Frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)