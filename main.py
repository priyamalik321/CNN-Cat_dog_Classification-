import numpy as np
import random
import matplotlib.pyplot as plt
import io
import base64
import yaml
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input

import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)

# Load file paths from config.yaml
def load_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load and preprocess the dataset
def load_data(config):
    # Load CSV paths from config
    x_train_path = config['data_paths']['x_train']
    y_train_path = config['data_paths']['y_train']
    x_test_path = config['data_paths']['x_test']
    y_test_path = config['data_paths']['y_test']

    x_train = np.loadtxt(x_train_path, delimiter=',')
    y_train = np.loadtxt(y_train_path, delimiter=',')
    x_test = np.loadtxt(x_test_path, delimiter=',')
    y_test = np.loadtxt(y_test_path, delimiter=',')

    # Reshape and normalize
    x_train = x_train.reshape(len(x_train), 100, 100, 3) / 255.0
    y_train = y_train.reshape(len(y_train), 1)
    x_test = x_test.reshape(len(x_test), 100, 100, 3) / 255.0
    y_test = y_test.reshape(len(y_test), 1)

    return x_train, y_train, x_test, y_test


# Create CNN model
def create_model():
    model = Sequential([
        Input(shape=(100, 100, 3)),  # Explicit Input Layer
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train the model
def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=5, batch_size=64)
    model.save('model/cnn_model.h5')


# Make a prediction on a random test image
def predict_random_image(model, x_test):
    idx = random.randint(0, len(x_test) - 1)
    image = x_test[idx, :].reshape(1, 100, 100, 3)

    prediction = model.predict(image)[0][0]
    label = 'cat' if prediction > 0.5 else 'dog'

    # Convert image to base64 for rendering in HTML
    img = x_test[idx, :]
    plt.imshow(img)
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return label, img_base64



@app.route('/')
def home():
    return render_template('index.html')





@app.route('/predict', methods=['POST'])
def predict():
    config = load_config()
    x_train, y_train, x_test, y_test = load_data(config)
    try:
        model = load_model('model/cnn_model.h5')
    except Exception as e:
        return f"Error loading model: {e}"

    label, img_base64 = predict_random_image(model, x_test)
    return render_template('result.html', label=label, img_data=img_base64)


if __name__ == '__main__':
    app.run(debug=True)
