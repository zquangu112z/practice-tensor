import numpy as np


def evaluateModel(test_sets_label, predicted_test):
    # print first $print_length items of parameter lists
    print_length = 1000
    print('predicted labels: ', predicted_test[:print_length])
    print('original labels: ', np.array(test_sets_label[:print_length]))

    # print errors
    print("there are ", len(test_sets_label),
          " test points. The errors are listed below:")

    # accuracy, precison, recall and number of positive cases
    TP = 0
    TP_FN = 0.00001
    TP_FP = 0.00001
    right = 0
    for i in range(len(test_sets_label)):
        if test_sets_label[i] == predicted_test[i]:
            right = right + 1
            if test_sets_label[i] != 0:
                TP = TP + 1
        else:
            print(predicted_test[i], ' ', np.array(test_sets_label[i]))
        if test_sets_label[i] != 0:
            TP_FN = TP_FN + 1
        if predicted_test[i] != 0:
            TP_FP = TP_FP + 1

    accuracy = right / len(test_sets_label)
    recall = TP / TP_FN
    precision = TP / TP_FP
    positiveCase = TP_FN
    print('Accuracy: ', accuracy)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('Positive case: ', positiveCase)
    return accuracy, recall, precision, positiveCase
