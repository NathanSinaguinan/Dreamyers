from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # Import Pydantic's BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
import string

app = FastAPI()

# Load the initial model
model = MultinomialNB()
vectorizer = TfidfVectorizer()

# Load the dataset
data = pd.read_csv('dataset copy.csv')  # Replace with your dataset file path

# Preprocess and vectorize the data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])
    
    # Tokenize the text and remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    # Rejoin tokens into a single string
    text = ' '.join(tokens)
    
    return text

data['TEXT'] = data['TEXT'].apply(preprocess_text)

# Train the initial model
X = vectorizer.fit_transform(data['TEXT'])
y = data['LABEL']
model.fit(X, y)

# Define a Pydantic model for the request payload
class TextPayload(BaseModel):
    text: str
    label: str

@app.post("/update_model")
async def update_model(new_data: TextPayload):
    """
    Update the model with new data.
    """
    try:
        # Extract and preprocess the new data
        new_text = preprocess_text(new_data.text)
        new_label = new_data.label
        
        # Add the new data to your dataset (data and labels)
        data['TEXT'] = data['TEXT'].append(pd.Series([new_text]), ignore_index=True)
        data['LABEL'] = data['LABEL'].append(pd.Series([new_label]), ignore_index=True)
        
        # Re-vectorize the entire dataset
        X = vectorizer.fit_transform(data['TEXT'])
        y = data['LABEL']
        
        # Retrain the model with the updated dataset
        model.fit(X, y)

        return {"message": "Model updated successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating the model: {str(e)}")

# Real-time prediction endpoint
@app.post("/predict")
async def predict(text_payload: TextPayload):
    """
    Make real-time predictions.
    """
    # Preprocess and vectorize the input text
    text = preprocess_text(text_payload.text)
    text_vectorized = vectorizer.transform([text])

    # Make a prediction
    prediction = model.predict(text_vectorized)

    return {"prediction": prediction[0]}

from fastapi.encoders import jsonable_encoder  # Import jsonable_encoder

@app.get("/evaluate")
async def evaluate_model():
    """
    Evaluate the model using a confusion matrix and classification report.
    """
    try:
        y_true = data['LABEL']
        y_pred = model.predict(X)

        # Calculate confusion matrix and classification report
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        # Convert the confusion matrix and report to JSON-serializable format
        cm_json = cm.tolist()  # Convert to list for JSON serialization
        report_json = jsonable_encoder(report)  # Use jsonable_encoder for complex data

        # Create a JSON-serializable response dictionary
        response_data = {
            "confusion_matrix": cm_json,
            "classification_report": report_json,
        }

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


if __name__ == "__main__":
    # Start the FastAPI app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
