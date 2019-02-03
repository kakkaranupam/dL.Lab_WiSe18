import airsim

import time
import os
import numpy as np

# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

import sys
import glob
import tensorflow as tf
import json
import pandas as pd
from Cooking import checkAndCreateDir
import h5py
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
from cnn_network import CNN

if ('../../PythonClient/' not in sys.path):
    sys.path.insert(0, '../../PythonClient/')
from AirSimClient import *

# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('model/models/*.h5') 
#     best_model = max(models, key=os.path.getctime)
#     MODEL_PATH = best_model
    
print('Using model {0} for testing.'.format(MODEL_PATH))
valid_loss = 0
#init = tf.global_variables_initializer()
test_loss = []
test_accuracy = []

#graph = tf.get_default_graph()

#pred = conv_net(x, weights, biases)
#correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#COOKED_DATA_DIR = 'cooked_data/'

#train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')
#state = train_dataset['image'][0:1]
#cnn = CNN()
#sess = tf.Session('', tf.Graph())
    
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
print('Connection established!')

car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

image_buf = np.zeros((1, 74, 255, 3))
#state_buf = np.zeros((1,4))


def get_image():
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    
    return image_rgba[70:144,0:255,0:3].astype(float)
    
saver = tf.train.import_meta_graph('SaveGraph/data-all.meta')
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    cnn = CNN()
    # print("SHAPEEEEEEE",(state.shape))
    
    #graph = tf.get_default_graph()
    #print("graph1",graph)
    
    saver.restore(sess, tf.train.latest_checkpoint('./SaveGraph'))
    graph = tf.get_default_graph()
    
    cnn.x = sess.graph.get_tensor_by_name('input:0')
    
    cnn.ypred = sess.graph.get_tensor_by_name('ypred:0')

    while (True):
        car_state = client.getCarState()

        if (car_state.speed < 5):
            car_controls.throttle = 1.0
        else:
            car_controls.throttle = 0.0

        image_buf[0] = get_image()
        # state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
#         model_output = model.predict([image_buf, state_buf])
        
        print("SSSSSSSSSSSSSSSSS")
        prediction = sess.run(cnn.ypred, feed_dict={cnn.x: image_buf})
        car_controls.steering = round(0.5 * float(prediction[0][0]), 2)

        print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))

        client.setCarControls(car_controls)
