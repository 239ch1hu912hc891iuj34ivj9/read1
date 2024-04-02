from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load pre-trained image classification model
model = keras.models.load_model("path_to_your_model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("file")
    results = []

    for file in uploaded_files:
        img = Image.open(file).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        label = np.argmax(predictions)
        confidence = np.max(predictions)
        
        results.append({'image': file.filename, 'label': label, 'confidence': confidence})

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
