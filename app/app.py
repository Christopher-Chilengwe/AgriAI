# app/app.py
from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load models
yield_model = joblib.load('crop_yield_model.pkl')
disease_model = tf.keras.models.load_model('plant_disease_cnn.h5')

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    data = request.json
    prediction = yield_model.predict([data['features']])
    return jsonify({'yield': prediction[0]})

@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    file = request.files['image']
    img = preprocess_image(file)
    prediction = disease_model.predict(img)
    return jsonify({'disease_class': int(np.argmax(prediction))})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)