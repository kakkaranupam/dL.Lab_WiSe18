import tensorflow as tf
import numpy as np


class CNN():
    """
    CNN Network class based on TensorFlow.
    """
    def __init__(self, lr=1e-4):
        self._build_model(lr)
        #self.state_dim = state_dim
        
    # Define Placeholders
    def _build_model(self,lr):
        self.x =        tf.placeholder(tf.float32, shape=[None, 74, 255, 3], name="input")
        print("X PLACEHOLDER DTYPE",tf.shape(self.x))
        self.actions_ = tf.placeholder(tf.int32, shape=[None,1])                  # Integer id of which action was selected
        self.targets_ = tf.placeholder(tf.float32,  shape=[None,1])

        #parameters of convolutional layer
        conv1_fmaps = 16
        conv1_ksize = 3
        conv1_stride = 1
        conv1_pad = "SAME"

        conv2_fmaps = 32
        conv2_ksize = 3
        conv2_stride = 1
        conv2_pad = "SAME"
        
        conv3_fmaps = 32
        conv3_ksize = 3
        conv3_stride = 1
        conv3_pad = "SAME"
        
        n_fc1 = 128
        n_outputs = 1
        
        self.conv1 = tf.layers.conv2d(self.x, filters=conv1_fmaps, kernel_size = conv1_ksize,
                         strides = conv1_stride, padding=conv1_pad,
                         activation = tf.nn.relu)
        print("C1", self.conv1)
        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2, padding='valid')
        print("P1", self.pool1)
        self.conv2 = tf.layers.conv2d(self.pool1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu)
        print(self.conv2)
        self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2, padding='valid')
        print(self.pool2)
        self.conv3 = tf.layers.conv2d(self.pool2, filters=conv3_fmaps, kernel_size=conv3_ksize,
                         strides=conv3_stride, padding=conv3_pad,
                         activation=tf.nn.relu)
        print("C3", self.conv3)
        self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=[2, 2], strides=2, padding='valid')
        print("P3", self.pool3)
        
        self.pool3_flat = tf.reshape(self.pool3, shape=[-1, conv3_fmaps * 9 * 31])
        print("P3 FLAT", self.pool3_flat)
        self.fc1 = tf.layers.dense(self.pool3_flat, n_fc1, activation = tf.nn.relu)
        print("FC1", self.fc1)
        self.predictions = tf.layers.dense(self.fc1, n_outputs)
        tf_ypred = tf.identity(self.predictions, "ypred")
        print("PRED ", self.predictions)
        
        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.x)[0]
        print("X XXXXXXXXXSHAPE",tf.shape(self.x))
        print("FFFFFFFFFFFFf", batch_size)
        #gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
        #self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # TODO: Loss and optimizer
        # ...
                
        # Calculate the loss
        self.losses = tf.squared_difference(self.targets_, self.predictions)
        self.loss = tf.reduce_mean(self.losses)
        tf_loss = tf.identity(self.loss, "yloss")

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss)
        
                
    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """

        if len(states.shape) < 4:
           states = np.expand_dims(states, axis=0)
        prediction = sess.run(self.predictions, { self.x: states })
        return prediction


    def update(self, sess, states, actions, targets):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.
        
        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        feed_dict = { self.x: states, self.targets_: targets, self.actions_: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
       
        return loss
    
    def loss_batch(self, states, sess, actions, targets):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.
        
        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        feed_dict = { self.x: states, self.targets_: targets, self.actions_: actions}
        # feed_dict = { self.targets_: targets, self.actions_: actions}
        loss = sess.run([self.loss], feed_dict)
       
        return loss
