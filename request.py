import requests

url = 'http://localhost:5000/results'
r = requests.post(url)

print(r.json())