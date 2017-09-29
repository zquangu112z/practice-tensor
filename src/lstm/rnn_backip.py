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
def onehot_to_index(arr):
    ret = []
    for i in range(len(arr)):
        row_labels = []
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                if arr[i][j][k] == 1:
                    row_labels.append(k)
                    break
        ret.append(row_labels)
    return ret

# get dataset using pickle
train_line_vectors, test_line_vectors, train_line_labels, test_line_labels = get_data_sets_pickle()
train_sets_label = onehot_to_index(train_line_labels)
# generate batch
# def nextBatch(number=100):
#     num_sequences = len(train_line_vectors)
#     i = random.randint(0, num_sequences - number)
#     batch_data = []
#     batch_labels = []
#     for i in range(number):
#         if len(train_line_vectors[i]) == max_seg_len:
#             batch_data.append(train_line_vectors[i])
#             batch_labels.append(train_sets_label[i])
#         elif len(train_line_vectors[i]) > max_seg_len:
#             batch_data.append(train_line_vectors[i][:max_seg_len])
#             batch_labels.append(train_sets_label[i][:max_seg_len])
#         else:
#             vector_len = len(train_line_vectors[i])
#             for i in range(0, max_seg_len - vector_len):
#                 train_line_vectors[i].append([0. for i in range(feature_size)])
#                 train_sets_label[i].append(19) # 19 for all adding labels
#             batch_data.append(train_line_vectors[i])

#     # print(batch_data[1])
#     # print(batch_labels[1])
#     return batch_data, batch_labels

# generate batch
def nextBatch(number=100):
    batch_data = train_line_vectors[:number]  # 100 sequences
    batch_labels = train_line_labels[:number]  

    feature_zero = [0.0 for i in range(feature_size)]
    

    for i in range(number):
        sequence_len = len(batch_data[i])
        if sequence_len == max_seg_len:
            pass
        elif sequence_len > max_seg_len:
            batch_data[i] = batch_data[i][:max_seg_len]
            batch_labels[i] = batch_labels[i][:max_seg_len]
        else:
            for i in range(max_seg_len - sequence_len):
                batch_data[i].append(feature_zero)
                batch_labels[i].append(19)
    return batch_data, batch_labels



num_batches = int(len(train_line_vectors)/batch_size)

# ----------------Training----------------
batchX_placeholder = tf.placeholder(
    tf.float32, [batch_size, max_seg_len, feature_size])
batchY_placeholder = tf.placeholder(
    tf.int32, [batch_size, max_seg_len])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W = tf.Variable(np.random.rand(state_size + feature_size, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((batch_size, state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((batch_size, num_classes)), dtype=tf.float32)

# unstack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward pass
current_state = init_state
print('current_state: ', current_state)
states_series = []

for current_input in inputs_series:   # unfold the graph
    print('---------------')
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
logits_series = [tf.matmul(state, W2) + b2 for state in states_series]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

# todo:modify this loss function, remove what added
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=labels) for logits, labels in zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)



# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     loss_list = []    
#     # _current_state = np.zeros(tf.shape(init_state))


#     for epoch_idx in range(num_epochs):
#         _current_state = np.zeros((batch_size, state_size))        
#         print(np.shape(_current_state))

#         batchX,batchY = nextBatch(batch_size)

#         print("New data, epoch", epoch_idx)



#         _total_loss, _train_step, _current_state, _predictions_series = sess.run(
#             [total_loss, train_step, current_state, predictions_series],
#             feed_dict={
#                 batchX_placeholder:batchX,
#                 batchY_placeholder:batchY,
#                 init_state:_current_state
#             })

#         loss_list.append(_total_loss)

#         if epoch_idx%100 == 0:
#             print("Epoch",epoch_idx, "Loss", _total_loss)
#                 # plot(loss_list, _predictions_series, batchX, batchY)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    for epoch_idx in range(num_epochs):
        _current_state = np.zeros((batch_size, state_size))
        # print('_current_state: ', _current_state.shape)
        # print('_current_state: ', init_state.shape)

        print("New data, epoch", epoch_idx)

        

        for batch_idx in range(num_batches):
            print('-----begin-----')

            batchX, batchY = nextBatch(batch_size)

            batchY = np.array(batchY)
            batchX = np.array(batchX)

            print("batchX", batchX.shape)
            print("batchX_placeholder", batchX_placeholder.shape)

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,


                    batchY_placeholder: batchY,

                    
                    init_state: _current_state
                })
            print('-----end-----')

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                # plot(loss_list, _predictions_series, batchX, batchY)

