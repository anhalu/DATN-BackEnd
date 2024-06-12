import io

import requests
import cv2
url = "http://0.0.0.0:8000/api/v1/file/upload"

img = cv2.imread('10.jpg')
_, img_encoded = cv2.imencode('.jpg', img)

# Convert the encoded image to bytes
image_bytes = img_encoded.tobytes()


files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}

headers = {
  'accept': 'application/json',
  'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0IiwiZXhwIjoxNzI0NDI2ODQ2fQ.crJ4mm0iHKkfSbQ8bI3qjzdax4Jsi_o29eiEzC2Cclo'
}

response = requests.request("POST", url, headers=headers, files=files)

print(response.text)
