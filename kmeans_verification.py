from kmeans import py_kmeans, cuda_kmeans
import matplotlib.pyplot as plt
import numpy as np
import time

plt.style.use('_mpl-gallery')

num_points = 10000
xs = []
ys = []
for (pos_x, pos_y) in [(-10, -10), (10, 10), (-10, 10), (10, -10)]:
    x = np.random.normal(pos_x, 3, size=num_points)
    y = np.random.normal(pos_y, 3, size=num_points)
    xs.append(x)
    ys.append(y)

x = np.hstack(xs)
y = np.hstack(ys)
vecs = np.vstack([x, y])
vecs = np.ascontiguousarray(vecs.transpose().astype(np.float32))

pre = time.time()
py_kmeans(vecs, 16, 100)
print(time.time() - pre)

pre = time.time()
ret, ret2 = cuda_kmeans(vecs, 16, 100)
print(time.time() - pre)

pre = time.time()
ret, ret2 = cuda_kmeans(vecs, 16, 100)
print(time.time() - pre)

# plt.scatter(x, y, c=0.3 * ret2)
# plt.savefig("imgs/kmeans_py.png", dpi=600)
