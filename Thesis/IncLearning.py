import requests
import pandas as pd

# Define the FastAPI server URL
server_url = "http://localhost:8000"  # Replace with your server's URL

# Load the CSV file
data = pd.read_csv('newdataset.csv')  # Replace with your CSV file path

# Loop through each row in the CSV file
for _, row in data.iterrows():
    # Extract text and label from the current row
    new_text = row["TEXT"]
    new_label = row["LABEL"]

    # Define the request payload with the "text" and "label" fields
    payload = {"text": new_text, "label": new_label}

    # Set the Content-Type header to indicate JSON data
    headers = {"Content-Type": "application/json"}

    # Send a POST request to the /update_model endpoint
    response = requests.post(f"{server_url}/update_model", json=payload, headers=headers)

    # Parse and print the response
    if response.status_code == 200:
        result = response.json()
        message = result["message"]
        print(f"Update Successful: {message}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
