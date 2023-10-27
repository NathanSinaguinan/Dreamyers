from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import string

app = FastAPI()

# Load the initial model
model = MultinomialNB()
vectorizer = TfidfVectorizer()

# Load the dataset
data = pd.read_csv('SMSDataset.csv', encoding='macroman')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])
    
    return text

data['TEXT'] = data['TEXT'].apply(preprocess_text)

# Train-Test Split
X = vectorizer.fit_transform(data['TEXT'])
y = data['LABEL']

# Use train-test split to separate data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the initial model on the resampled data
model.fit(X_train, y_train)

# Define a Pydantic model for the request payload
class TextPayload(BaseModel):
    text: str
    label: str

@app.post("/update_model")
async def update_model(new_data: TextPayload):
    try:
        global data  # Declare 'data' as a global variable
        # Extract and preprocess the new data
        new_text = preprocess_text(new_data.text)
        new_label = new_data.label

        # Create a new DataFrame for the new data
        new_row = pd.DataFrame({"TEXT": [new_text], "LABEL": [new_label]})

        # Concatenate the new data to the existing data
        data = pd.concat([data, new_row], ignore_index=True)
        
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
        # Create a new vectorizer for evaluation and fit it to the evaluation data
        evaluation_vectorizer = TfidfVectorizer()
        X_test_vectorized = evaluation_vectorizer.fit_transform(data['TEXT'])  # Vectorize the evaluation data

        y_true = data['LABEL']
        y_pred = model.predict(X_test_vectorized)

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
