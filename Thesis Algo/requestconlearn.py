import requests

url = "http://127.0.0.1:8000/continuous_learning/"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}
data = [
    {"text": "Congratulations! You've won a free vacation.", "label": "spam"},
    {"text": "Hi, how are you doing today?", "label": "ham"},
    {"text": "Urgent: Your package is delayed.", "label": "spam"},
]

response = requests.post(url, headers=headers, json=data)
print(response.json())