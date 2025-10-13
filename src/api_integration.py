import requests

# Define the URL with the fruit name (replace '[fruit-name]' with the actual fruit name)
url = 'https://www.fruityvice.com/api/fruit/[fruit-name]'
# url = 'https://www.fruityvice.com/api/fruit/apple'

# Make the API request and handle the response
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    #get only nutrition facts
    print(data["nutritions"])
else:
    print("API request failed with status code:", response.status_code)