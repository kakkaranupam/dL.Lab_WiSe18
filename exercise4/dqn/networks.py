import tensorflow as tf
import numpy as np

# TODO: add your Convolutional Neural Network for the CarRacing environment.

class CNN():
    """
    CNN Network class based on TensorFlow.
    """
    def __init__(self, lr=1e-4):
        self._build_model(lr)
        #self.state_dim = state_dim
        
    # Define Placeholders
    def _build_model(self,lr):
        self.x =        tf.placeholder(tf.float32, shape=[None, 96, 96, 5])
        self.actions_ = tf.placeholder(tf.int32, shape=[None])                  # Integer id of which action was selected
        self.targets_ = tf.placeholder(tf.float32,  shape=[None])

        #parameters of convolutional layer
        conv1_fmaps = 32
        conv1_ksize = 3
        conv1_stride = 1
        conv1_pad = "SAME"

        conv2_fmaps = 64
        conv2_ksize = 3
        conv2_stride = 1
        conv2_pad = "SAME"
        
        conv3_fmaps = 128
        conv3_ksize = 3
        conv3_stride = 1
        conv3_pad = "SAME"
        
        n_fc1 = 128
        n_outputs = 5
        
        self.conv1 = tf.layers.conv2d(self.x, filters=conv1_fmaps, kernel_size = conv1_ksize,
                         strides = conv1_stride, padding=conv1_pad,
                         activation = tf.nn.relu)
        
        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2, padding='valid')
        
        self.conv2 = tf.layers.conv2d(self.pool1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu)
        
        self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2, padding='valid')
        
        self.conv3 = tf.layers.conv2d(self.pool2, filters=conv3_fmaps, kernel_size=conv3_ksize,
                         strides=conv3_stride, padding=conv3_pad,
                         activation=tf.nn.relu)
        
        self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=[2, 2], strides=2, padding='valid')
       
        self.pool3_flat = tf.reshape(self.pool3, shape=[-1,conv3_fmaps*12*12])
        
        self.fc1 = tf.layers.dense(self.pool3_flat, n_fc1, activation = tf.nn.relu)
        
        self.predictions = tf.layers.dense(self.fc1, n_outputs)
       
        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.x)[0]
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # TODO: Loss and optimizer
        # ...
                
        # Calculate the loss
        self.losses = tf.squared_difference(self.targets_, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

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

class CNNTargetNetwork(CNN):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """
    def __init__(self, lr=1e-4, tau=0.01):
        CNN.__init__(self, lr)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx,var in enumerate(tf_vars[0:total_vars//2]):
            op_holder.append(tf_vars[idx+total_vars//2].assign(
              (var.value()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
        return op_holder
      
    def update(self, sess):
        for op in self._associate:
          sess.run(op)


class NeuralNetwork():
    """
    Neural Network class based on TensorFlow.
    """
    def __init__(self, state_dim, num_actions, hidden=20, lr=5e-4):
        self._build_model(state_dim, num_actions, hidden, lr)
        self.state_dim = state_dim
        
    def _build_model(self, state_dim, num_actions, hidden, lr):
        """
        This method creates a neural network with two hidden fully connected layers and 20 neurons each. The output layer
        has #a neurons, where #a is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with a learning rate of 1e-4).
        """
        
        self.states_ = tf.placeholder(tf.float32, shape=[None, state_dim])
        self.actions_ = tf.placeholder(tf.int32, shape=[None])                  # Integer id of which action was selected
        self.targets_ = tf.placeholder(tf.float32,  shape=[None])               # The TD target value

        # network
        fc1 = tf.layers.dense(self.states_, hidden, tf.nn.relu)
        fc2 = tf.layers.dense(fc1, hidden, tf.nn.relu)
        self.predictions = tf.layers.dense(fc2, num_actions)

        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.states_)[0]
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.targets_, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

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
        states = states.reshape(-1, self.state_dim)
        prediction = sess.run(self.predictions, { self.states_: states })
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
        feed_dict = { self.states_: states, self.targets_: targets, self.actions_: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class TargetNetwork(NeuralNetwork):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """
    def __init__(self, state_dim, num_actions, hidden=20, lr=5e-4, tau=0.01):
        NeuralNetwork.__init__(self, state_dim, num_actions, hidden, lr)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx,var in enumerate(tf_vars[0:total_vars//2]):
            op_holder.append(tf_vars[idx+total_vars//2].assign(
              (var.value()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
        return op_holder
      
    def update(self, sess):
        for op in self._associate:
          sess.run(op)
