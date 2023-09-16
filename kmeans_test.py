from kmeans import py_kmeans, cuda_kmeans
import numpy as np
import time


n = 100000
d = 100
vecs = np.random.normal(0, 10, size=(n, d)).astype(np.float32)
pre = time.time()
py_kmeans(vecs, 16, 10)
print(time.time() - pre)


pre = time.time()
ret, ret2 = cuda_kmeans(vecs, 16, 10)
print(time.time() - pre)

# plt.scatter(x, y, c=0.3 * ret2)
# plt.savefig("imgs/kmeans_py.png", dpi=600)
