import numpy as np
from itertools import chain
from fii_mde.utils import evaluateModel, log, get_data_minibatch, get_test_sets
import timeit
import tensorflow as tf
from fii_mde.utils import evaluateModel
from sklearn.metrics import confusion_matrix

DEEP_MODEL_DIR = "build/model/MyDeepNN.ckpt"


def next_batch(batch_size, batch_idx):
    data_sets = get_data_minibatch(
        batch_size=batch_size, start_idx=136 + batch_idx * batch_size)
    train_sets = data_sets.train

    train_line_vectors = train_sets.vectors
    train_line_labels = train_sets.labels
    # print(train_line_labels.shape)
    # flatting 2-D to 1-D
    train_sets_vectors = np.array(sum(train_line_vectors, []))
    # log("Feature demesion: ", len(train_sets_vectors[0]))
    train_sets_label = list(chain.from_iterable(train_line_labels))
    return train_sets_vectors, train_sets_label


class MyDeepNN:

    def __init__(self):
        self.feature_size = 9414 + 260  # 9414 local feature
        # self.feature_size = 9414 + 100  # 9414 local feature
        self.state_size = 100
        self.classes_size = 19
        self.batch_size = 200
        self.num_epochs = 20

        self.num_datapoint = 3700  # CHANGE
        self.testdataProportion = 0.1

        self.numBatches = int(self.num_datapoint *
                              (1 - self.testdataProportion) / self.batch_size)

    def prepare(self):
        x = tf.placeholder(tf.float32, [None, self.feature_size])

        W1 = tf.Variable(tf.random_normal(
            [self.feature_size, self.state_size]))
        b1 = tf.Variable(tf.random_normal([self.state_size]))

        W2 = tf.Variable(tf.random_normal(
            [self.state_size, self.classes_size]))
        b2 = tf.Variable(tf.random_normal([self.classes_size]))

        state = tf.nn.tanh(tf.matmul(x, W1) + b1)

        y = tf.nn.softmax(tf.matmul(state, W2) + b2)

        y_ = tf.placeholder(
            tf.float32, [None, self.classes_size])  # correct answer

        return x, y_, y

    def train(self):
        x, y_, y = self.prepare()
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                                      tf.log(y), reduction_indices=[1]))

        train_step = tf.train.AdamOptimizer(
            0.05).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            log('Init variable')
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.num_epochs):
                for batch_idx in range(self.numBatches):
                    # get mini-batch
                    # todo kiem tra ham next_batch
                    batch_xs, batch_ys = next_batch(
                        batch_size=self.batch_size, batch_idx=batch_idx)
                    _train_step, _accuracy = sess.run([train_step, accuracy], feed_dict={
                        x: batch_xs, y_: batch_ys})
                    if batch_idx % 5 == 0:
                        print('EPOCH ', epoch, ': ', _accuracy)
            try:
                saver.save(sess, DEEP_MODEL_DIR)
                print('Saved model')
            except:
                print("Can't save trained model.")
            print("Training finish.")

    def predict(self):
        data_sets = get_test_sets(num_datapoint=self.num_datapoint,
                                  testdataProportion=self.testdataProportion)
        test_sets = data_sets.test

        test_line_vectors = test_sets.vectors
        test_line_labels = test_sets.labels
        test_sets_vectors = np.array(sum(test_line_vectors, []))
        test_sets_label = list(chain.from_iterable(test_line_labels))

        x, y_, y = self.prepare()
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, DEEP_MODEL_DIR)

            log("Prediction")
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            prediction = tf.argmax(y, 1)
            result = tf.argmax(y_, 1)

            accuracy_, prediction_,result_ = session.run([accuracy, prediction,result], feed_dict={
                x: test_sets_vectors, y_: test_sets_label})
            evaluateModel(result_, prediction_)

if __name__ == '__main__':
    myDeepNN = MyDeepNN()
    # myDeepNN.train()
    myDeepNN.predict()
