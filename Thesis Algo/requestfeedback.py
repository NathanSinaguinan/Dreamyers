import requests
from pydantic import BaseModel
from typing import List, Optional

# Model for Feedback Data
class FeedbackData(BaseModel):
    message_id: str
    true_label: str

# Sample data for feedback
feedback_data = [
    FeedbackData(message_id="1", true_label="ham"),
    FeedbackData(message_id="2", true_label="spam"),
    # Add more feedback data if needed
]

# Convert FeedbackData objects to dictionaries
feedback_data_dicts = [item.dict() for item in feedback_data]

# Send feedback to the FastAPI server
url = "http://localhost:8000/feedback/"  # Change the URL if needed
response = requests.post(url, json=feedback_data_dicts)

print(response.json())
