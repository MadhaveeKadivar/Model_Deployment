import requests 

resp = requests.post("http://localhost:5000/api/predict", files={'file': open('seven.png', 'rb')})

print(resp.text)