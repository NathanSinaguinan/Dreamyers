import requests

# Define the FastAPI server URL
server_url = "http://localhost:8000"  # Replace with your server's URL

# Text sample you want to classify
text_to_classify = "Hello, kamusta ka? sana okay ka lang dyan sainyo"

# Define the request payload with both "text" and "label" fields
payload = {"text": text_to_classify, "label": ""}  # Set the label as needed

# Set the Content-Type header to indicate JSON data
headers = {"Content-Type": "application/json"}

# Send a POST request to the /predict endpoint
response = requests.post(f"{server_url}/predict", json=payload, headers=headers)

# Parse and print the response
if response.status_code == 200:
    result = response.json()
    predicted_label = result["prediction"]
    print(f"Predicted Label: {predicted_label}")
else:
    print(f"Error: {response.status_code} - {response.text}")
