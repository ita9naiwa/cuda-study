import numpy as np

from matsum import cuda_vec_sum
d = 100000000
a = np.ones(d).astype(np.float32)
b = np.ones(d).astype(np.float32)

c = cuda_vec_sum(a, b)

print((c - 2).sum())
