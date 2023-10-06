import requests

# Define the FastAPI server URL
server_url = "http://localhost:8000"  # Replace with your server's URL

# Text sample you want to classify
text_to_classify = "Good Morning Sir, inform ko lng po kayo na forfeited na po yung unit nyo since wla po kayo paramamdam upupurged na po nmiin today yung unit since angtagal na po ksi sain nito wla po kakyong prmdam thank you If willing pa po kyo kunin yung pwde n,n po kaso need nyo magbyad ng storage fee 50 per day. From 1 month up to now. Thank you"

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
