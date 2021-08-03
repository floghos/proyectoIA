import numpy as np

import os

for i in range (5):
    print(i)
    arr = [k+i for k in range(10)]
    np.save('experiments/test', arr)