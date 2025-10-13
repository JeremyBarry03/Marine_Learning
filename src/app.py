from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

class_names = ["apple", "apricot", "banana", "blueberry", "cactus-fruit", "cantaloupe", "cherry", "dates",
               "grape", "grapefruit", "guava", "kiwi", "lemon", "lime", "lychee", "mango", "orange",
               "peach", "pear", "pineapple", "plum", "pomegranate", "raspberry", "strawberry", 
               "tomato", "watermelon"]

model = tf.keras.models.load_model('../models/optimized-model.h5')

@app.route('/')
def home():
    return "Hello, this is your Flask backend!"

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    img = img.resize((64, 64)) 
    img_array = np.array(img) / 255.0

    predictions = model.predict(np.expand_dims(img_array, axis=0))
    score = tf.nn.softmax(predictions[0])
    fruit_name = class_names[np.argmax(score)]

    #make the API request and handle the response
    url = f"https://www.fruityvice.com/api/fruit/{fruit_name}"
    response = requests.get(url)
    nutrition = ""

    if response.status_code == 200:
        data = response.json()
        #get only nutrition facts
        nutrition = data["nutritions"]
        #print(data["nutritions"])
    else:
        print("API request failed with status code:", response.status_code)

    prediction = ("This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 1000 * np.max(score)))

    return jsonify({'prediction': prediction, 'nutrition': nutrition})

if __name__ == '__main__':
    app.run(debug=True)
