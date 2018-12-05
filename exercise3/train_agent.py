from __future__ import print_function

import gzip
import os
import pickle
from datetime import datetime

import tensorflow as tf

from model import Model
from tensorboard_evaluation import Evaluation
from utils import *


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    # data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    # DATA 2 : 45K
    data_file = os.path.join(datasets_dir, 'data2.pkl.gzip')

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) * n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) * n_samples):]

    # X_train, y_train = X[500:2000], y[500:2000]
    # X_valid, y_valid = X[:500], y[:500]

    return X_train, y_train, X_valid, y_valid


def minibatched(data, batch_size):
    assert len(data) % batch_size == 0, ("Data length {} is not multiple of batch size {}"
                                         .format(len(data), batch_size))
    return data.reshape(-1, batch_size, *data.shape[1:])


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):
    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    # X_train = np.expand_dims(rgb2gray(X_train), axis=3)
    X_train_new = []

    # divide in batches and convert
    bs = 1000
    for batch in range(X_train.shape[0] // bs):
        batch_x = X_train[batch * bs:min((batch + 1) * bs, len(X_train))]
        # print("BATCH X shape --- ", batch_x.shape)
        batch_x = np.expand_dims(rgb2gray(batch_x), axis=3)
        # print("BATCH X EXPAND DIMS shape --- ", batch_x.shape)
        X_train_new[batch * bs:min((batch + 1) * bs, len(X_train))] = batch_x
    X_train_new = np.array(X_train_new)
    # print("XTRAIN NEW SHAPE --- ", X_train_new.shape)
    # print("YTR ", y_train)

    y_n = np.zeros(y_train.shape[0])
    for i in range(y_train.shape[0]):
        y_n[i] = action_to_id(y_train[i])
        # if y_n[i] == 4:
            # print("YN TRAIN --- >>>", y_n[i])
    # y_train = action_to_id_all(y_train)

    y_train = one_hot_encoding(y_n, 5)

    print("y_train.shape ... ", y_train.shape)

    # fname = "./results_bc_agent-%s.txt" % datetime.now().strftime("%Y%m%d-%H%M%S")
    # fh = open(fname, "w")
    # for i in range(y_train.shape[0]):
    #     fh.write(str(i) + " -- " + str(y_train[i]) + "\n")
    # fh.close()

    X_valid = np.expand_dims(rgb2gray(X_valid), axis=3)

    y_n = np.zeros(y_valid.shape[0])
    for i in range(y_valid.shape[0]):
        y_n[i] = action_to_id(y_valid[i])
        # if y_n[i] == 4:
        #     print("YN VALID --- >>>", y_n[i])

    # y_valid = action_to_id_all(y_valid)
    y_valid = one_hot_encoding(y_n, 5)
    return X_train_new, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    # TODO: specify your neural network in model.py
    agent = Model(lr, model_dir)
    tensorboard_eval = Evaluation(tensorboard_dir)
    batch_size = X_train.shape[0] // n_minibatches

    print("BATCH SIZE ... ", batch_size)
    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # saver = tf.train.import_meta_graph('SaveGraph/data-all.meta')

    saver = tf.train.import_meta_graph('models/savedG.meta')
    saver.restore(agent.sess, tf.train.latest_checkpoint('./models'))
    graph = tf.get_default_graph()

    print("GRAPH RESTORED ... ", graph)
    x = agent.sess.graph.get_tensor_by_name('input:0')
    y = agent.sess.graph.get_tensor_by_name('output:0')
    cost = agent.sess.graph.get_tensor_by_name('cost:0')
    optimizer = agent.sess.graph.get_operation_by_name('optimizer')
    accuracy = agent.sess.graph.get_tensor_by_name('accuracy:0')
    ypred = agent.sess.graph.get_tensor_by_name('ypred:0')
    writeSummary = tf.summary.FileWriter('./output', agent.sess.graph)
    print("TENSORS RESTORED ... ", x, y, cost, optimizer, accuracy, ypred)

    epoch = 1
    for epo in range(epoch):
        print("EPOCH EPOCH EPOCH EPOCH ----------------- >>>>>>>>>>>>>>>>", epo)
        count = 0
        tr_cost = 0
        tr_acc = 0
        va_acc = 0
        trc = []
        tra = []
        vaa = []
        eval_dict = {"tr_cost": tr_cost, "tr_acc": tr_acc, "va_acc": va_acc}

        for batch in range(X_train.shape[0] // batch_size):
            batch_x = X_train[batch * batch_size:min((batch + 1) * batch_size, len(X_train))]
            batch_y = y_train[batch * batch_size:min((batch + 1) * batch_size, len(y_train))]

            print("BATCH VALUE ... ", batch)
            print("BATCH X SHAPE ... ", batch_x.shape)
            print("BATCH Y SHAPE ... ", batch_y.shape)

            run_opt = agent.sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            tr_cost, tr_acc = agent.sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            trc.append(tr_cost)
            tra.append(tr_acc)
            # eval_dict = {"tr_cost": tr_cost, "tr_acc": tr_acc, "va_acc": 0}
            eval_dict = {"tr_cost": tr_cost, "tr_acc": tr_acc, "va_acc": va_acc}
            tensorboard_eval.write_episode_data(batch, eval_dict)

        for batch in range(X_valid.shape[0] // batch_size):
            batch_x = X_valid[batch * batch_size:min((batch + 1) * batch_size, len(X_valid))]
            batch_y = y_valid[batch * batch_size:min((batch + 1) * batch_size, len(y_valid))]

            yhat = agent.sess.run(ypred, feed_dict={x: batch_x, y: batch_y})
            count += len(yhat[yhat is True])

            print("YHAT ... ", type(yhat))
            print("COUNT OF CORRECT YHAT ... ", count)

            # accuracy = tf.reduce_mean(tf.cast(yhat, tf.float32))
            loss, acc = agent.sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            vaa.append(acc)
            print(" VALIDATION ACCURACY ... ", acc)
            # eval_dict = {"tr_cost": 0, "tr_acc": 0, "va_acc": va_acc}
            eval_dict = {"tr_cost": tr_acc, "tr_acc": tr_acc, "va_acc": va_acc}
            tensorboard_eval.write_episode_data(batch, eval_dict)

        va_acc = count / len(X_valid)

        print(" TRAINING LOSS = " + "{:.6f}".format(tr_cost) + ", TRAINING ACCURACY = " + "{:.6f}".format(tr_acc))
        # print(" VALIDATION ACCURACY ... ", va_acc)

        # print(" LOSS ... ", loss, " --- ACCURACY ... ", acc)
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
        tensorboard_eval.write_episode_data(epo, eval_dict)

    # TODO: save your agent
    agent.save(os.path.join(model_dir, "agent.ckpt"))
    agent.sess.close()


if __name__ == "__main__":
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    print("X TRAIN shape", X_train.shape)
    print("y TRAIN shape", y_train.shape)
    print("X VALID shape", X_valid.shape)
    print("y VALID shape", y_valid.shape)

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    print("PRE PROCESSING DONE")
    print("X TRAIN shape", X_train.shape)
    print("y TRAIN shape", y_train.shape)
    print("X VALID shape", X_valid.shape)
    print("y VALID shape", y_valid.shape)

    # train_model(X_train, y_train, X_valid, n_minibatches=100, batch_size=64, lr=0.001)
    # Different Minibatch Setup
    train_model(X_train, y_train, X_valid, n_minibatches=250, batch_size=64, lr=0.025)
