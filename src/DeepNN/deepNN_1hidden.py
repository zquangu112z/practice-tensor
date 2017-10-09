import numpy as np
from itertools import chain
from fii_mde.utils import evaluateModel, log, get_data_minibatch, get_test_sets, get_data_sets
import timeit
import tensorflow as tf
from fii_mde.utils import evaluateModel
from fii_mde.utils import db_context
import pickle
from sklearn.utils import shuffle



DEEP_MODEL_DIR = "build/model/MyDeepNN.ckpt"
VEC_DIR = "build/trainvecs.pkl"
LABEL_DIR = "build/trainlabel.pkl"
def dense_to_one_hot(labels_dense, num_classes=6):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    index = list(index_offset + labels_dense.ravel())
    labels_one_hot.flat[index] = 1
    return labels_one_hot


def one_hot_transform(vector):

    labels = []
    for label in vector:
        label = np.array(label)
        labels.append(dense_to_one_hot(label))

    return labels
def getData(num_datapoint=20000):
    db = db_context()
    cursor = db.cursor()

    query = "SELECT vector, label, value FROM FEATURE LIMIT " + str(num_datapoint) + ";"
    cursor.execute(query)
    data = cursor.fetchall()

    data = np.array(data)
    return data

class MyDeepNN:
    def __init__(self):
        self.feature_size = 260
        self.state_size = 300
        self.classes_size = 6
        self.batch_size = 100  # CHANGE
        self.num_epochs = 1500

        self.num_datapoint = 20000  # CHANGE
        self.testdataProportion = 0.2  # CHANGE

        self.tol = 0.03
        # self.accuracies = [0, 0]
        self.losses = [500, 500, 500, 500]

        self.numBatches = int(self.num_datapoint *
                              (1 - self.testdataProportion) / self.batch_size)
        
        try:
            with open(VEC_DIR, 'rb') as f:
                self.train_sets_vectors = pickle.load(f)
            with open(LABEL_DIR, 'rb') as f:
                self.train_sets_label = pickle.load(f)
        except:
            data = getData(self.num_datapoint)
            
            self.train_sets_vectors = np.array(sum(data[:, 0], []))
            self.train_sets_label = np.array(list(chain.from_iterable(np.array(one_hot_transform(data[:, 1])))))
            
            with open(VEC_DIR, 'wb') as f:
                pickle.dump(self.train_sets_vectors, f)
            with open(LABEL_DIR, 'wb') as f:
                pickle.dump(self.train_sets_label, f)
        print("Num words: ", len(self.train_sets_vectors))

    def shuffleData(self):
        self.train_sets_vectors, self.train_sets_label = shuffle(self.train_sets_vectors, self.train_sets_label, random_state=0)

    def next_batch(self, batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = batch_idx * self.batch_size + self.batch_size
        return self.train_sets_vectors[start_idx:end_idx], self.train_sets_label[start_idx:end_idx]

    def prepare(self):
        x = tf.placeholder(tf.float32, [None, self.feature_size])

        W1 = tf.Variable(tf.random_normal(
            [self.feature_size, self.state_size]))
        b1 = tf.Variable(tf.random_normal([self.state_size]))

        W2 = tf.Variable(tf.random_normal(
            [self.state_size, self.classes_size]))
        b2 = tf.Variable(tf.random_normal([self.classes_size]))

        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

        # y = tf.nn.softmax(tf.matmul(hidden1, W2) + b2)
        y = tf.matmul(hidden1, W2) + b2

        y_ = tf.placeholder(
            tf.float32, [None, self.classes_size])  # correct answer

        return x, y_, y

    def beginTrain(self, sess, train_step, accuracy, x, y_,loss):
        for epoch in range(self.num_epochs):
            self.shuffleData()
            _accuracies = 0
            _losses = 0
            for batch_idx in range(self.numBatches):
                batch_xs, batch_ys = self.next_batch(batch_idx)
                _train_step, _accuracy, _loss= sess.run([train_step, accuracy, loss], feed_dict={
                    x: batch_xs, y_: batch_ys})

                _accuracies += _accuracy
                _losses += _loss

            # khong cai thien dc model nua
            _accuracies = _accuracies / self.numBatches
            _losses = _losses / self.numBatches
            # if (_accuracies - self.accuracies[0]) < self.tol and (_accuracies - self.accuracies[1]) < self.tol:
            #     print("Early stoped!")
            #     return sess
            # self.accuracies[0] = self.accuracies[1]
            # self.accuracies[1] = _accuracies
            print('EPOCH ', epoch, ': ', _accuracies, _losses)
            if _losses < self.tol and self.losses[0] < self.tol and self.losses[1] < self.tol and self.losses[2] < self.tol and self.losses[3] < self.tol:
                return sess
            self.losses[0] = self.losses[1]
            self.losses[1] = self.losses[2]
            self.losses[2] = self.losses[3]
            self.losses[3] = _losses

        return sess

    def train(self):
        print('numBatches: ', self.numBatches)
        x, y_, y = self.prepare()
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
        # tf.log(y), reduction_indices=[1]))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        train_step = tf.train.AdamOptimizer(0.002).minimize(loss)
        # train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

        # correct_prediction = tf.nn.in_top_k(y, y_, 1)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            log('Init variable')
            sess.run(tf.global_variables_initializer())
            sess = self.beginTrain(sess, train_step, accuracy, x, y_,loss)

            try:
                saver.save(sess, DEEP_MODEL_DIR)
                print('Saved model')
            except Exception as e:
                print("Can't save trained model: ", e)
            print("Training finish.")

    def predict(self):
        test_sets = get_test_sets()
        test_sets_vectors = np.array(sum(test_sets.test.vectors, []))
        test_sets_label = list(chain.from_iterable(test_sets.test.labels))

        x, y_, y = self.prepare()
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, DEEP_MODEL_DIR)

            log("Prediction")
            # y = tf.nn.softmax(y)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            prediction = tf.argmax(y, 1)
            result = tf.argmax(y_, 1)

            accuracy_, prediction_, result_ = session.run([accuracy, prediction, result], feed_dict={
                x: test_sets_vectors, y_: test_sets_label})
            evaluateModel(result_, prediction_)

if __name__ == '__main__':
    myDeepNN = MyDeepNN()
    # myDeepNN.train()
    myDeepNN.predict()

