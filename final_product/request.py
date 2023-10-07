import requests

url = 'https://spotcom.vercel.app/'
r = requests.post(url)

print(r.json())
