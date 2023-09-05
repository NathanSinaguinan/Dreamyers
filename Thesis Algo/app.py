from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
from typing import List, Optional

# Create the FastAPI app
app = FastAPI()

# Load the initial model and vectorizer or create a new one if not available
try:
    model_and_vectorizer = joblib.load("SmishingDetection.joblib")
    model, vectorizer = model_and_vectorizer
except FileNotFoundError:
    model = MultinomialNB()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Function to preprocess and vectorize the text
def preprocess_text(text):
    text = text.lower().replace('[^\w\s]', '')
    return " ".join(text.split())

# Function to update the model with new data
def update_model(new_data: List['ContinuousLearningData']):
    new_texts = [preprocess_text(data.text) for data in new_data]
    new_labels = [data.label for data in new_data if data.label is not None]

    if not new_labels:
        # If no true labels are provided, there is nothing to update.
        return

    X_new = vectorizer.transform(new_texts).toarray()
    model.partial_fit(X_new, new_labels, classes=np.unique(new_labels))

# Endpoint to receive new messages and return predictions
class MessageInput(BaseModel):
    text: str
    label: Optional[str] = None  # Make label optional

@app.post("/predict/")
async def predict_message(messages: List[MessageInput]):
    # Preprocess and vectorize the new messages
    new_texts = [preprocess_text(message.text) for message in messages]
    X_new = vectorizer.transform(new_texts).toarray()

    # Make predictions
    predicted_labels = model.predict(X_new)

    # Create a JSONResponse object with the predicted labels
    response = JSONResponse(content={"predictions": predicted_labels.tolist()})

    return response

# Endpoint to provide feedback on the model's predictions
from requestfeedback import FeedbackData

@app.post("/feedback/")
async def send_feedback(feedback_data: List[FeedbackData]):
    # Extract feedback data from the request
    message_ids = [item.message_id for item in feedback_data]
    true_labels = [item.true_label for item in feedback_data]

    # Update the model with the feedback data
    update_model(feedback_data)

    # Save the updated model and vectorizer together
    model_and_vectorizer = (model, vectorizer)
    joblib.dump(model_and_vectorizer, "SmishingDetection.joblib")

    # Return a simple acknowledgment JSON response
    acknowledgment = {"message": "Feedback received. Model updated with new data."}

    return {"message": "Feedback received. Model updated with new data."}

# Model for Continuous Learning Data
class ContinuousLearningData(BaseModel):
    text: str
    label: Optional[str] = None  # Make label optional

@app.post("/continuous_learning/")
async def continuous_learning(data: List[ContinuousLearningData]):
    # Process the data and extract texts
    new_texts = [preprocess_text(item.text) for item in data]

    # If a label is provided, update the model with the data
    if any(item.label for item in data):
        update_model(data)

    # Preprocess and vectorize the new messages
    X_new = vectorizer.transform(new_texts).toarray()

    # Make predictions
    predicted_labels = model.predict(X_new)

    # Return only the predicted labels in the response
    response = JSONResponse(content={"predictions": predicted_labels.tolist()})

    return response
