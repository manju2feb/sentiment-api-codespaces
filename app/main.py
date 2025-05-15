from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextInput(BaseModel):
    text: str

# Load model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'sentiment_model.pkl')
try:
    model = joblib.load(model_path)
    logger.info(f"✅ Loaded model from {model_path}")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    raise

@app.get("/")
def read_root():
    logger.info("Health check endpoint accessed.")
    return {"message": "Welcome to Sentiment Analysis API"}

@app.post("/predict/")
def predict_sentiment(input: TextInput):
    logger.info(f"📨 Prediction request received: {input.text}")
    try:
        prediction = model.predict([input.text])[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        logger.info(f"✅ Prediction: {sentiment}")
        return {"text": input.text, "sentiment": sentiment}
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")
