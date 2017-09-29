# khong hoi tu, co the sai o ham loss
import numpy as np
from itertools import chain
import pickle
from src.utils import log, get_data_sets_pickle
import timeit
import tensorflow as tf
import random

# ----------------parameters of model----------------
num_epochs = 50
state_size = 100
num_classes = 19
max_seg_len = 40  # so tu toi da trong 1 cau
batch_size = 100
feature_size = 3099


# ----------------Data preparing----------------
# get dataset using pickle
train_line_vectors, test_line_vectors, train_line_labels, test_line_labels = get_data_sets_pickle()

# generate batch


def nextBatch(train_line_vectors, train_line_labels, batchSize, batchIdx):
    # start_index = random.randint(
    #     1, len(train_line_vectors) - batchSize - 2)   # 1 vs -2 de cho chac'
    start_index = batchIdx * batchSize

    batch_data = np.array(
        train_line_vectors[start_index:start_index + batchSize])  # 100 sequences
    batch_labels = np.array(
        train_line_labels[start_index:start_index + batchSize])

    feature_zero = np.array([0.0 for i in range(feature_size)])
    label_zero = np.array([0.0 for i in range(num_classes)])
    label_zero[0] = 1.0

    batch_data_ret = []
    batch_labels_ret = []
    seg_len_ret = []

    for i in range(batchSize):
        sequence_len = len(batch_data[i])
        if sequence_len == max_seg_len:
            batch_data[i] = np.array(batch_data[i])
            batch_labels[i] = np.array(batch_labels[i])

            batch_data_ret.append(batch_data[i].tolist())
            batch_labels_ret.append(batch_labels[i].tolist())
            seg_len_ret.append(sequence_len)
        elif sequence_len > max_seg_len:
            batch_data[i] = np.array(batch_data[i][:max_seg_len])
            batch_labels[i] = np.array(batch_labels[i][:max_seg_len])

            batch_data[i].reshape([max_seg_len, 3099])
            batch_labels[i].reshape([max_seg_len, 19])

            batch_data_ret.append(batch_data[i].tolist())
            batch_labels_ret.append(batch_labels[i].tolist())
            seg_len_ret.append(max_seg_len)
        else:
            for j in range(max_seg_len - sequence_len):
                batch_data[i] = np.append(
                    batch_data[i], [batch_data[i][sequence_len - 1]], axis=0)
                batch_labels[i] = np.append(
                    batch_labels[i], [batch_labels[i][sequence_len - 1]], axis=0)
            batch_data[i].reshape([max_seg_len, 3099])
            batch_labels[i].reshape([max_seg_len, 19])

            batch_data_ret.append(batch_data[i].tolist())
            batch_labels_ret.append(batch_labels[i].tolist())
            seg_len_ret.append(sequence_len)

    batch_data = np.array(batch_data)
    return np.array(batch_data_ret), np.array(batch_labels_ret), np.array(seg_len_ret)


num_batches = int(len(train_line_vectors) / batch_size)

# ----------------Training----------------
init_state = tf.placeholder(tf.float32, [None, state_size])
batchX_placeholder = tf.placeholder(
    tf.float32, [None, max_seg_len, feature_size])
batchY_placeholder = tf.placeholder(
    tf.int32, [None, max_seg_len, num_classes])

batchSegLen_placeholder = tf.placeholder(
    tf.int32, [None])

W = tf.Variable(np.random.rand(
    state_size + feature_size, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((batch_size, state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((batch_size, num_classes)), dtype=tf.float32)

# unstack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward pass
current_state = init_state
# print('current_state: ', current_state)
states_series = []

for current_input in inputs_series:   # unfold the graph
    current_input = tf.reshape(current_input, [batch_size, feature_size])
    # print('current_input: ', current_input)
    input_and_state_concatenated = tf.concat(
        [current_input, current_state], 1)  # Increasing number of columns
    # Broadcasted addition
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)
    states_series.append(next_state)
    current_state = next_state

# Broadcasted addition
logits_series = [tf.matmul(state, W2) +
                 b2 for state in states_series]    # (20,100,19)

predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
print('predictions_series: ', len(predictions_series))
print('predictions_series[0]: ', predictions_series[0].shape)
print('predictions_series[1]: ', predictions_series[1].shape)
print('predictions_series[1][0]: ', predictions_series[1][0])


losses = [tf.nn.softmax_cross_entropy_with_logits(
logits=logits, labels=labels) for logits, labels in zip(logits_series,
labels_series)]   # 20 ,100,1

losses_unstack = tf.unstack(losses, axis=1)   # 100,20,1
print('losses_unstack: ', len(losses_unstack))
print('losses_unstack[0]: ', losses_unstack[0].shape)
print('losses_unstack[1]: ', losses_unstack[1].shape)
print('losses_unstack[1][0]: ', losses_unstack[1][0])

total_losses = []
for i , batch_losses in enumerate(losses_unstack):
    # print('batchSegLen_placeholder[', i,']: ', batchSegLen_placeholder[i])
    total_losses.append(tf.reduce_mean(batch_losses[:batchSegLen_placeholder[i]]))

print('total_losse: ', len(total_losses))
total_loss = tf.reduce_mean(total_losses)
train_step = tf.train.AdagradOptimizer(0.01).minimize(total_loss)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    try: 
        saver.restore(sess, 'model/mde.ckpt')
    except:
        loss_list = []
        _current_state = np.zeros((batch_size, state_size))

        for epoch_idx in range(num_epochs):

            print("New data, epoch", epoch_idx)

            # for batch_idx in range(100):   # ------------------------------------
            for batch_idx in range(num_batches):
                # print('-----begin-----')

                batchX, batchY, batchSegLen = nextBatch(
                    train_line_vectors, train_line_labels, batch_size, batch_idx)

                _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                    [total_loss, train_step, current_state, predictions_series],
                    feed_dict={
                        init_state: _current_state,
                        batchX_placeholder: batchX,
                        batchY_placeholder: batchY,
                        batchSegLen_placeholder: batchSegLen
                    })
                # print('-----end-----')

                loss_list.append(_total_loss)

                if batch_idx % 10 == 0:
                    print("Step", batch_idx, "Loss", _total_loss)

        save_path = saver.save(sess, "model/rnnmde.ckpt")
        log("Model saved in file: %s" % save_path)

    # predict
    # batchX, batchY, batchSegLen = nextBatch(
    #     test_line_vectors, test_line_labels, 10, 0)

    # _predictions_series = sess.run(
    #             [predictions_series],
    #             feed_dict={
    #                 batchX_placeholder: batchX,
    #                 batchY_placeholder: batchY,
    #                 batchSegLen_placeholder: batchSegLen
    #             })

    # predictions_series = [tf.argmax(logits) for logits in logits_series]
    # sess.run(predictions_series)


