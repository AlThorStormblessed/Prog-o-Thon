import requests

url = 'https://prog-o-thon-fmx9.vercel.app/'
r = requests.post(url)

print(r.json())
