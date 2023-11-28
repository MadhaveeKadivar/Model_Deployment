import requests 

resp = requests.post("http://127.0.0.1:5000/api/predict", files={'file': open('three.png', 'rb')})

print(resp.text)