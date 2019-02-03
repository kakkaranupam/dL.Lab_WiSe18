import tensorflow as tf
import json
import os
import numpy as np
import pandas as pd
from Cooking import checkAndCreateDir
import h5py
from tensorboard_evaluation import *
from cnn_network import CNN
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# << The directory containing the cooked data from the previous step >>
COOKED_DATA_DIR = 'cooked_data_03/'
# COOKED_DATA_DIR = 'cooked_data_02/'
# COOKED_DATA_DIR = 'cooked_data_OLD01/'

train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')
eval_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'eval.h5'), 'r')
test_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'test.h5'), 'r')

num_train_examples = train_dataset['image'].shape[0]
num_eval_examples = eval_dataset['image'].shape[0]
num_test_examples = test_dataset['image'].shape[0]
print("TRSN", train_dataset['image'])
print("TRSN", num_eval_examples)
print("TRSN", num_test_examples)
batch_size=128
truefalse = np.array([True, False])
shuff = np.random.choice(truefalse, int(num_train_examples))
obj = CNN()
tensorboard_dir="./tensorboard"
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
#tensorboard_dir_eval="./tensorboardEval"
#if not os.path.exists(tensorboard_dir_eval):
#    os.makedirs(tensorboard_dir_eval)
tensorboard = Evaluation(tensorboard_dir, ["Training_loss", "Evaluation_loss"])
# tensorboard_eval = Evaluation(tensorboard_dir_eval, ["Evaluation_loss"])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #train_loss = []
    print("... train agent")
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    epochs = 5
    x_train = train_dataset['image']
    y_train = train_dataset['label']
    x_valid = eval_dataset['image']
    y_valid = eval_dataset['label']
    for i in range(epochs):
        print("Epoch----->",i)
        tloss = 0
        print(num_train_examples)
        print(batch_size)
        shuff = np.random.choice(truefalse, int(num_train_examples))
        for batch in range(num_train_examples//batch_size):
            print("HERE IN Training")
            
            batch_x = x_train[batch*batch_size:min((batch+1)*batch_size,len(x_train))]
            batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))]
            if(shuff[batch]):
                batch_x, batch_y = shuffle(batch_x, batch_y, random_state=0)
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            #opt = sess.run(optimizer, feed_dict={x: batch_x,
            #                                              y: batch_y})
            #loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
            #                                              y: batch_y})
            predicted_out = obj.predict(sess, batch_x)
            #print(predicted_out)
            print("SHAPE PRED",batch_x.shape)
            tloss = obj.update(sess=sess, states=batch_x, actions=predicted_out, targets=batch_y)
            # print("Iter " + str(i) + ", TLoss= " + "{:.6f}".format(tloss) + ", predicted_out= " + "{:.6f}".format(predicted_out[0]))
            print("Iter " + str(i) + ", TLoss= " + "{:.6f}".format(tloss))
        vloss = 0
        for batch in range(num_eval_examples//batch_size):
            print("HERE IN Validation")
            batch_x = x_valid[batch*batch_size:min((batch+1)*batch_size,len(x_valid))]
            batch_y = y_valid[batch*batch_size:min((batch+1)*batch_size,len(y_valid))]
            if(shuff[batch]):
                batch_x, batch_y = shuffle(batch_x, batch_y, random_state=0)
            predicted_out = obj.predict(sess, batch_x)
            vloss = obj.loss_batch(sess=sess, states=batch_x, actions=predicted_out, targets=batch_y)
            print("Iter " + str(i) + ", VLoss= " + "{:.6f}".format(vloss[0]))
        tensorboard.write_episode_data(i, eval_dict={ "Training_loss" : tloss, "Evaluation_loss" : vloss[0] })
        
        #for batch in range(len(x_valid)//batch_size):
        #    batch_x = x_valid[batch*batch_size:min((batch+1)*batch_size,len(x_valid))]
        #    batch_y = y_valid[batch*batch_size:min((batch+1)*batch_size,len(y_valid))]    
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
         #   cp = sess.run(correct_prediction, feed_dict={x: batch_x,
         #                                                 y: batch_y})

          #  count += len(cp[cp==True])
          #  print("CorEEECT PRED", type(cp))

           # print("CorEEECT PRED count in a batch", count)

            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                         # y: batch_y})

        #valid_acc = count/len(x_valid)
        #+ ", Training Accuracy= " + \
        #          "{:.5f}".format(acc)
        #print("Validation Accuracy :",valid_acc)
        print("Optimization Finished!")
        #train_loss.append(tloss)
#         train_accuracy.append(acc)
        # Calculate accuracy for all 10000 mnist test images
        #lc = dict()
        #lc["accuracy"] = acc
        #lc["loss"] = loss
        #lc["global_step"] = i
        #learning_curve.append(lc)
        summary_writer.close()
    all_saver = tf.train.Saver()
    all_saver.save(sess, './SaveGraph/data-all')



#ans = (optimizer.graph == tf.get_default_graph())
#print("=========================>>>>",ans)



#model = dict()
#model = {"wts":weights,"bias":biases,"Prediction":pred,"CorrectP":correct_prediction}
#return learning_curve, model  # TODO: Return the validation error after each epoch (i.e learning curve) and your model
