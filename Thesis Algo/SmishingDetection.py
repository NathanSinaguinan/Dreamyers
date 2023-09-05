import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

def preprocess_text(text):
    text = text.lower().replace('[^\w\s]', '')
    return " ".join(text.split())

def train_initial_model(vectorizer):
    # Load your initial dataset into a pandas DataFrame
    # Assuming your initial dataset is stored in a CSV file
    data = pd.read_csv('dataset copy.csv')

    # Step 1: Data Preprocessing
    data['TEXT'] = data['TEXT'].apply(preprocess_text)

    # Step 2: TF-IDF Representation
    X = vectorizer.fit_transform(data['TEXT']).toarray()

    # Step 3: Data Split
    y = data['LABEL']

    # Step 4: Naive Bayes Classifier
    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer

def update_model(model, vectorizer, new_texts, new_labels):
    # Convert to TF-IDF representation
    X_new = vectorizer.transform(new_texts).toarray()

    # Get all possible classes (including the ones from the initial training)
    all_classes = np.union1d(model.classes_, np.unique(new_labels))

    # Update the model with the new data, explicitly setting the classes parameter
    model.partial_fit(X_new, new_labels, classes=all_classes)

    return model, vectorizer

def evaluate_model(model, vectorizer, test_data, test_labels):
    # Preprocess the test data
    test_texts = test_data.apply(preprocess_text)

    # Convert to TF-IDF representation
    X_test = vectorizer.transform(test_texts).toarray()

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, y_pred)

    # Generate classification report
    class_report = classification_report(test_labels, y_pred)

    return accuracy, class_report

if __name__ == "__main__":
    # Load the existing model and vectorizer or create a new one if not available
    try:
        model_and_vectorizer = joblib.load("SmishingDetection.joblib")
        model, vectorizer = model_and_vectorizer
    except FileNotFoundError:
        model = None
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    # Train the initial model if it doesn't exist
    if model is None:
        model, vectorizer = train_initial_model(vectorizer)
    else:
        # Load new labeled data for continuous learning (replace this with your actual new dataset)
        new_data = pd.read_csv('new_labeled_data.csv')
        new_texts = new_data['TEXT']
        new_labels = new_data['LABEL']

        # Print the new texts and new labels to verify the data
        #print("New Texts:")
        #print(new_texts)
        #print("New Labels:")
        #print(new_labels)

        # Update the model with the new data
        model, vectorizer = update_model(model, vectorizer, new_texts, new_labels)

    # Load the initial dataset for evaluation
    test_data = pd.read_csv('dataset copy.csv')['TEXT']
    test_labels = pd.read_csv('dataset copy.csv')['LABEL']
    accuracy, class_report = evaluate_model(model, vectorizer, test_data, test_labels)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(class_report)

    # Save the updated model and vectorizer together
    model_and_vectorizer = (model, vectorizer)
    joblib.dump(model_and_vectorizer, "SmishingDetection.joblib")
