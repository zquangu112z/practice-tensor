# Da chay dc roi, can cai tien ham loss de loai tru nhung truong hop added


import numpy as np
from itertools import chain
import pickle
from src.utils import log, get_data_sets_pickle
import timeit
import tensorflow as tf
import random

# ----------------parameters of model----------------
num_epochs = 50
state_size = 4
num_classes = 19
max_seg_len = 20  # so tu toi da trong 1 cau
batch_size = 100
feature_size = 3099


# ----------------Data preparing----------------
# get dataset using pickle
train_line_vectors, test_line_vectors, train_line_labels, test_line_labels = get_data_sets_pickle()

# generate batch


def nextBatch(number=100):
    start_index = random.randint(
        1, len(train_line_vectors) - number - 2)   # 1 vs -2 de cho chac'
    batch_data = np.array(
        train_line_vectors[start_index:start_index + number])  # 100 sequences
    batch_labels = np.array(
        train_line_labels[start_index:start_index + number])

    feature_zero = np.array([0.0 for i in range(feature_size)])
    label_zero = np.array([0.0 for i in range(num_classes)])
    label_zero[0] = 1.0

    batch_data_ret = []
    batch_labels_ret = []

    for i in range(number):
        sequence_len = len(batch_data[i])
        # print(batch_data[i].shape)
        if sequence_len == max_seg_len:
            batch_data[i] = np.array(batch_data[i])
            batch_labels[i] = np.array(batch_labels[i])

            batch_data_ret.append(batch_data[i].tolist())
            batch_labels_ret.append(batch_labels[i].tolist())
            # log('pass', i)
        elif sequence_len > max_seg_len:
            # log('> ', i)
            # print('----', len(batch_data[i]))
            batch_data[i] = np.array(batch_data[i][:max_seg_len])
            # print(len(batch_data[i]))
            batch_labels[i] = np.array(batch_labels[i][:max_seg_len])

            batch_data[i].reshape([20, 3099])
            batch_labels[i].reshape([20, 19])

            batch_data_ret.append(batch_data[i].tolist())
            batch_labels_ret.append(batch_labels[i].tolist())
        else:

            # print('<<<', len(batch_data[i]))
            for j in range(max_seg_len - sequence_len):
                batch_data[i] = np.append(
                    batch_data[i], [feature_zero], axis=0)
                batch_labels[i] = np.append(
                    batch_labels[i], [label_zero], axis=0)
            batch_data[i].reshape([20, 3099])
            batch_labels[i].reshape([20, 19])

            batch_data_ret.append(batch_data[i].tolist())
            batch_labels_ret.append(batch_labels[i].tolist())
            # print('>>>', len(batch_data[i]))
        # sequence_len = len(batch_data[i])
        # print(sequence_len)

    batch_data = np.array(batch_data)

    # print('batch_data.shape: ', np.array(batch_data_ret).shape)
    # for i in batch_data:
    #     print(i.shape)
    # print(batch_data[0].shape)
    # print(batch_data[1].shape)
    # print(batch_labels[0][0].shape)
    return np.array(batch_data_ret), np.array(batch_labels_ret)


num_batches = int(len(train_line_vectors) / batch_size)

# ----------------Training----------------
init_state = tf.placeholder(tf.float32, [batch_size, state_size])
batchX_placeholder = tf.placeholder(
    tf.float32, [batch_size, max_seg_len, feature_size])
batchY_placeholder = tf.placeholder(
    tf.int32, [batch_size, max_seg_len, num_classes])


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
    # print('input_and_state_concatenated: ', input_and_state_concatenated)

    # Broadcasted addition
    # print(W)
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)
    states_series.append(next_state)
    # print('next_state: ', next_state)
    # print('init_state: ', init_state)
    current_state = next_state

# Broadcasted addition
logits_series = [tf.matmul(state, W2) +
                 b2 for state in states_series]    # (20,100,19)
# print('logits_series: ', len(logits_series))
# print('logits_series[0]: ', logits_series[0].shape)
# print('logits_series[0][0]: ', logits_series[0][0].shape)

# predictions_series = [[tf.nn.softmax(logit) for logit in logits] for logits in logits_series]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
# print('predictions_series: ', len(predictions_series))
# print('predictions_series[0]: ', predictions_series[0].shape)
# print('predictions_series[0][0]: ', predictions_series[0][0].shape)

# todo:modify this loss function, remove what added
losses = [tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=labels) for logits, labels in zip(logits_series, labels_series)]
# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
# logits=logits, labels=labels) for logits, labels in zip(logits_series,
# labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    for epoch_idx in range(num_epochs):
        _current_state = np.zeros((batch_size, state_size))
        # print('_current_state: ', _current_state.shape)
        # print('_current_state: ', init_state.shape)

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            # print('-----begin-----')

            batchX, batchY = nextBatch()

            # print("batchX", batchX.shape)
            # print("batchY", batchY.shape)
            # for i in range(len(batchX)):
            #     try:
            #         print("batchX_", i, batchX[i].shape)
            #     except:
            #         print(len(batchX[i]))
            #         print(len(batchX[i][0]))
            # print("batchX_1", batchX[1].shape)
            # print("batchX_placeholder", batchX_placeholder.shape)

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    init_state: _current_state,
                    batchX_placeholder: batchX,


                    batchY_placeholder: batchY



                })
            # print('-----end-----')

            loss_list.append(_total_loss)

            if batch_idx % 10 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                # plot(loss_list, _predictions_series, batchX, batchY)
