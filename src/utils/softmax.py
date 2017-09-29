scores = [3.0, 1.0, 0.2]

import numpy as np
import math 

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print(softmax(scores))

    