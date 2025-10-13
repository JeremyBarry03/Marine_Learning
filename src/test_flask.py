import requests

# URL of your Flask backend
backend_url = 'http://127.0.0.1:5000/'

# Path to the image file you want to test
image_path = '../data/test/apple/1_apple.jpg'

# Open the image file and send it to the /predict endpoint
with open(image_path, 'rb') as file:
    files = {'file': file}
    response = requests.post(f'{backend_url}/predict', files=files)

# Print the response
print(response.status_code)
print(response.content)
