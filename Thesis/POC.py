import requests
import pandas as pd

# Define the FastAPI server URL
server_url = "http://localhost:8000"  # Replace with your server's URL

# Function to predict a label
def predict_label():
    text_to_classify = input("Enter the text you want to classify: ")
    payload = {"text": text_to_classify, "label": ""}
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{server_url}/predict", json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        predicted_label = result["prediction"]
        print(f"Predicted Label: {predicted_label}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Function to add data to the model
def add_data_to_model():
    csv_file_path = input("Enter the path to the CSV file containing data: ")
    data = pd.read_csv(csv_file_path)

    for _, row in data.iterrows():
        new_text = row["TEXT"]
        new_label = row["LABEL"]

        payload = {
            "text": new_text,
            "label": new_label
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{server_url}/update_model", json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            message = result["message"]
            print(f"Update Successful: {message}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

# Prompt the user for their choice
while True:
    print("Choose an option:")
    print("1. Predict a label")
    print("2. Add data to the model")
    print("3. Exit")
    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        predict_label()
    elif choice == "2":
        add_data_to_model()
    elif choice == "3":
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
