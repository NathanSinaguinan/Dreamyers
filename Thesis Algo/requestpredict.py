import requests

url = "http://127.0.0.1:8000/predict/"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}
data = [
    {"text": "Your password requires resetting click here to reset your bank password!"},
    {"text": "We would just like to get you attention to participate in the forum to keep up the community!"},
]

response = requests.post(url, headers=headers, json=data)
print(response.json())
