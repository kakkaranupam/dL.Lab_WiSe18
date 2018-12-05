import tensorflow as tf
import os


def conv2d(x, weight, bias, strides=1):
    # Conv2D, Bias, ReLU
    x = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)


def maxpool2d(x, kernel=2):
    return tf.nn.max_pool(x, ksize=[1, kernel, kernel, 1], strides=[1, kernel, kernel, 1], padding='SAME')


def conv_net(x, weights, biases):
    # BUILD the Conv Net

    # Conv Layer 1, Max Pool 1
    convlayer1 = conv2d(x, weights['convW1'], biases['convB1'])
    print("convlayer1 ... ", convlayer1.shape)
    maxplayer1 = maxpool2d(convlayer1, kernel=2)
    print("mplayer1 ... ", maxplayer1.shape)

    # Conv Layer 2, Max Pool 2
    convlayer2 = conv2d(maxplayer1, weights['convW2'], biases['convB2'])
    print("convlayer2 ... ", convlayer2.shape)
    maxplayer2 = maxpool2d(convlayer2, kernel=2)
    print("mplayer2 ... ", maxplayer2.shape)

    # Conv Layer 3, Max Pool 3
    convlayer3 = conv2d(maxplayer2, weights['convW3'], biases['convB3'])
    print("convlayer3 ... ", convlayer3.shape)
    maxplayer3 = maxpool2d(convlayer3, kernel=2)
    print("mplayer3 ... ", maxplayer3.shape)

    # Dense Layer with ReLU
    mp3flatten = tf.reshape(maxplayer3, [-1, weights['densW'].get_shape().as_list()[0]])
    print("mp3flatten ... ", mp3flatten.shape)
    denselayer = tf.nn.relu(tf.add(tf.matmul(mp3flatten, weights['densW']), biases['densB']))
    print("denselayer ... ", denselayer.shape)

    # Output Layer
    outlayer = tf.add(tf.matmul(denselayer, weights['outW']), biases['outB'])
    print("outlayer ... ", outlayer.shape)
    return outlayer


class Model:

    def __init__(self, lr, num_layers=3, model_dir="./models"):
        # TODO: Define network
        # ...

        self.lr = lr

        # Define Placeholders
        x = tf.placeholder(tf.float32, [None, 96, 96, 1], name="input")
        y = tf.placeholder(tf.float32, [None, 5], name="output")

        weights = {
            'convW1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
            'convW2': tf.get_variable('W1', shape=(3, 3, 32, 64), initializer=tf.contrib.layers.xavier_initializer()),
            'convW3': tf.get_variable('W2', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),
            'densW': tf.get_variable('W3', shape=(12 * 12 * 128, 128), initializer=tf.contrib.layers.xavier_initializer()),
            'outW': tf.get_variable('W6', shape=(128, 5), initializer=tf.contrib.layers.xavier_initializer()),
        }
        biases = {
            'convB1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            'convB2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'convB3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'densB': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'outB': tf.get_variable('B4', shape=(5), initializer=tf.contrib.layers.xavier_initializer()),
        }

        logitslayer = conv_net(x, weights, biases)
        tf_logitslayer = tf.identity(logitslayer, "logitslayer")
        print("LOGITSLAYER ... ", logitslayer)

        # TODO: Loss and optimizer
        # ...
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logitslayer, labels=y))
        tf_cost = tf.identity(cost, "cost")
        print("MODEL - COST ... ", cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost, name="optimizer")
        print("MODEL - OPTIM ...", optimizer)

        ypred = tf.equal(tf.argmax(logitslayer, 1), tf.argmax(y, 1))
        tf_ypred = tf.identity(ypred, "ypred")
        print("MODEL - YPRED ... ", ypred)

        accuracy = tf.reduce_mean(tf.cast(ypred, tf.float32))
        tf_accu = tf.identity(accuracy, "accuracy")
        print("MODEL - ACCURACY ... ", accuracy)

        init = tf.global_variables_initializer()

        # TODO: Start tensorflow session
        # ...

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)
        self.save(os.path.join(model_dir, "savedG"))

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
