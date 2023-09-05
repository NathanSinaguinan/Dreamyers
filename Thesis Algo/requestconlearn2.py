import requests

url = "http://127.0.0.1:8000/continuous_learning/"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}
data = [
    {"text": "Register now! Free 178 pesos upon registering to our website, Visit here!"},
    {"text": "Hi, How are you?"},
    {"text": "Poker Thrills 3percent GCASH bonus! Bising website.pyc"},
]

response = requests.post(url, headers=headers, json=data)
print(response.json())