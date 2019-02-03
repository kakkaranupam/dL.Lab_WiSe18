 #!/usr/bin/env python -W ignore::DeprecationWarning
import numpy as np
import pandas as pd
import h5py
from PIL import Image, ImageDraw
import os
import Cooking
import random
import tensorflow as tf
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import os
import cv2
from math import floor, ceil, pi
import scipy.misc
%matplotlib inline

# << Point this to the directory containing the raw data >>
RAW_DATA_DIR = 'raw_data/'

# PREPROCESSED_DATA
PREPROCESSED_DATA_DIR = 'preprocessed_data/'

# << Point this to the desired output directory for the cooked (.h5) data >>
COOKED_DATA_DIR = 'cooked_data/'

# The folders to search for data under RAW_DATA_DIR
# DATA_FOLDERS = ['normal_1', 'normal_2', 'normal_3', 'normal_4']
DATA_FOLDERS = []

# The size of the figures in this notebook
FIGURE_SIZE = (10,10)

for root, dirs, files in os.walk(RAW_DATA_DIR):
    DATA_FOLDERS = [d for d in dirs]
    break
print(DATA_FOLDERS)

full_path_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in DATA_FOLDERS]

dataframes = []
for folder in full_path_raw_folders:
    current_dataframe = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')
    current_dataframe['Folder'] = folder
    dataframes.append(current_dataframe)
    
dataset = pd.concat(dataframes, axis=0)

print('Number of data points: {0}'.format(dataset.shape[0]))

# dataset.head()

def get_image_paths():
    all_png = []
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.endswith(".png"):
                all_png.append(os.path.join(root, file).replace('\\', '/'))
    return all_png

def tf_resize_images(X_img_file_paths):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (144, 256), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Each image is resized individually as different image may be of different size.
        for index, file_path in enumerate(X_img_file_paths):
            img = mpimg.imread(file_path)[:, :, :3] # Do not read alpha channel.
            resized_img = sess.run(tf_img, feed_dict = {X: img})
            X_data.append(resized_img)

    X_data = np.array(X_data, dtype = np.float32) # Convert to numpy
    return X_data

def clip_images(X_imgs):
#     points = [(1,76), (1,135), (255,135), (255,76)]
    X_cropped = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (144, 256, 3))
    tf_img1 = tf.image.crop_to_bounding_box(X, 70, 0, 74, 255)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            clipped_imgs = sess.run([tf_img1], feed_dict = {X: img})
            X_cropped.extend(clipped_imgs)
    X_cropped = np.array(X_cropped)
    return X_cropped

def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (74, 255, 3))
    tf_img1 = tf.image.flip_left_right(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
#     print(X_flip)
    X_flip = np.array(X_flip)
    return X_flip

X_png = get_image_paths()
# print(X_png)
clipped_dir = PREPROCESSED_DATA_DIR + 'clipped_images/'
flipped_dir = PREPROCESSED_DATA_DIR + 'flipped_images/'
name_f = 'flipped_'
name_c = 'clipped_'

if not os.path.exists(clipped_dir):
    os.makedirs(clipped_dir)
    os.makedirs(clipped_dir + 'images/')
if not os.path.exists(flipped_dir):
    os.makedirs(flipped_dir)
    os.makedirs(flipped_dir + 'images/')

img_names = [x.split('/')[-1] for x in X_png]
print(len(img_names))
print(len(dataset.index))
ds2 = dataset[dataset.ImageFile.isin(img_names)]
print(len(ds2.index))

c_df = []
f_df = []
batch_size = 1000
num_batches = len(X_png)//batch_size

for batch in range(num_batches):
    X_png_batch = X_png[batch * batch_size:min((batch + 1) * batch_size, len(X_png))]
    X_imgs_batch = tf_resize_images(X_png_batch)
    
    img_names = [x.split('/')[-1] for x in X_png_batch]
    clipped_images = clip_images(X_imgs_batch)
    print(clipped_images.shape)
    for i, f in enumerate(clipped_images):
        scipy.misc.imsave(clipped_dir + 'images/' + name_c + img_names[i], f)
    print("BATCH # ", batch, " ... CLIPPED WRITTEN ... ")
    cdf = ds2[ds2.ImageFile.isin(img_names)]
    print("len(cdf.index)", len(cdf.index))
    cdf.drop('ImageFile', axis = 1, inplace = True)
    cdf.insert(loc=15, column='ImageFile', value=[name_c + i for i in img_names])
    cdf.drop('Folder', axis = 1, inplace = True)
    cdf.insert(loc=16, column='Folder', value=clipped_dir)
    c_df.append(cdf)
    
    lc = len(clipped_images)
    k = lc * 45 // 100
    print(k)
    indices = random.sample(range(lc), k)
    indices = sorted(indices)
    img_names = [img_names[i] for i in indices]
    flipped_images = flip_images(clipped_images[indices])
    print(flipped_images.shape)
    for i, f in enumerate(flipped_images):
        scipy.misc.imsave(flipped_dir + 'images/' + name_f + img_names[i], f)
    print("BATCH # ", batch, " ... FLIPPED WRITTEN ... ")
    fdf = ds2[ds2.ImageFile.isin(img_names)]
    print("len(fdf.index)", len(fdf.index))
    steering = []
    for steer in fdf.Steering:
        if(steer != 0.0):
            steering.append(-1 * steer)
        else:
            steering.append(steer)
    fdf.drop('Steering', axis = 1, inplace = True)
    fdf.insert(loc=9, column='Steering', value=steering)
    fdf.drop('ImageFile', axis = 1, inplace = True)
    fdf.insert(loc=15, column='ImageFile', value=[name_f + i for i in img_names])
    fdf.drop('Folder', axis = 1, inplace = True)
    fdf.insert(loc=16, column='Folder', value=clipped_dir)
    f_df.append(fdf)

c_dataset = pd.concat(c_df, axis=0)
f_dataset = pd.concat(f_df, axis=0)

c_dataset.to_csv(os.path.join(clipped_dir, 'airsim_rec.txt'), sep='\t')
f_dataset.to_csv(os.path.join(flipped_dir, 'airsim_rec.txt'), sep='\t')

train_eval_test_split = [0.7, 0.2, 0.1]
full_path_pp_folders = [clipped_dir, flipped_dir]
print(full_path_pp_folders)
Cooking.cook(full_path_pp_folders, COOKED_DATA_DIR, train_eval_test_split)