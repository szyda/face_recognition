import requests
import base64

url = ""

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

image_path = "../../database/identity.jpeg"
base64_image = encode_image_to_base64(image_path)

payload = {
    "name": "name",
    "image": base64_image
}

response = requests.post(url, json=payload)

print(response.json())
