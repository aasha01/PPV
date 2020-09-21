from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

#Image utils
from tensorflow import keras
import cv2


# Define a flask app
app = Flask(__name__)

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Model saved with Keras model.save()
MODEL_PATH = 'models/Model_Predict_HumanFace.h5'
BS = 256

# Load your trained model
loadedModel = keras.models.load_model(MODEL_PATH)
adam = keras.optimizers.Adam(lr = 0.001)
loadedModel.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print('Model loaded and compiled with Adam optimizer. Start serving...')

# Check https://keras.io/applications/
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_con = image.reshape((1, 224, 224, 3))
    image_con = np.array(image_con) / 255.0
    predIdxs = loadedModel.predict(image_con, batch_size=BS)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    kv = {0: 'airplane', 1: 'car', 2: 'cat', 3: 'dog', 4: 'flower', 5: 'fruit', 6: 'motorbike', 7: 'person'}

    l = dict((k, v) for k, v in kv.items())
    prednames = l[predIdxs[0]]
    print(prednames)
    return prednames


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        basepath = os.path.dirname(__file__)

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        prednames = model_predict(file_path)

        '''mydir = os.path.join(basepath, 'uploads')
        # !/usr/bin/python
        print(mydir)
        filelist = [f for f in os.listdir(mydir) if f.endswith(".jpg")]
        for f in filelist:
            try:
                print(os.path.join(mydir, f))
                os.remove(os.path.join(mydir, f))
            except Exception as e:
                print(e)
            else:
                print("%s removed" % (os.path.join(mydir, f)))'''

        return prednames
    return None


if __name__ == '__main__':
    #app.run( debug = True, threaded = True)
    #In Cloud
    app.run(host='0.0.0.0', debug = True, threaded = True)

    # Serve the app with gevent
    #http_server = WSGIServer(('0.0.0.0', 5000), app)
    #http_server.serve_forever()