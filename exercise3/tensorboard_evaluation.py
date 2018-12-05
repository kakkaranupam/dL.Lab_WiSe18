import tensorflow as tf


class Evaluation:

    def __init__(self, store_dir):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.tf_writer = tf.summary.FileWriter(store_dir)

        self.tr_cost = tf.placeholder(tf.float32, name="tr_cost")
        tf.summary.scalar("tr_cost", self.tr_cost)

        # TODO: define more metrics you want to plot during training (e.g. training/validation accuracy)
        self.tr_acc = tf.placeholder(tf.float32, name="tr_acc")
        tf.summary.scalar("tr_acc", self.tr_acc)

        self.va_acc = tf.placeholder(tf.float32, name="va_acc")
        tf.summary.scalar("va_acc", self.va_acc)

        self.performance_summaries = tf.summary.merge_all()

    def write_episode_data(self, episode, eval_dict):
        # TODO: add more metrics to the summary
        # summary = self.sess.run(self.performance_summaries, feed_dict={self.tf_loss : eval_dict["loss"]})
        summary = self.sess.run(self.performance_summaries, \
                                feed_dict={self.tr_cost: eval_dict["tr_cost"],
                                           self.tr_acc: eval_dict["tr_acc"],
                                           self.va_acc: eval_dict["va_acc"]})
        self.tf_writer.add_summary(summary, episode)
        self.tf_writer.flush()

    def close_session(self):
        self.tf_writer.close()
        self.sess.close()
