cimport cython
from cython.parallel cimport prange
from cython.cimports.libc.stdlib import malloc, free
from libc.string cimport memset
import numpy as np
cimport numpy as cnp


cdef extern from "../src/kmeans.h":
    float cu_kmeans(float *vecs, int n, int d, int K, int iter,
                    float *o_centroids, int *o_belonging)


def cuda_kmeans(float[:, :] vecs, int K, int iters):
    cdef int n = vecs.shape[0]
    cdef int d = vecs.shape[1]
    o_centroids = np.zeros((K, d), dtype=np.float32)
    o_belonging = np.zeros(n, dtype=np.int32)
    cdef float[:, :] o_centroid_view = o_centroids
    cdef int[:] o_belonging_view = o_belonging
    cu_kmeans(
        &vecs[0, 0], n, d, K, iters,
        &o_centroid_view[0, 0],
        &o_belonging_view[0]
    )
    return o_centroids, o_belonging

@cython.cdivision(True)
@cython.boundscheck(False)
def py_kmeans(float[:, :] vecs, int K, int iters):
    cdef int epoch, i, j, k,  curr_idx
    cdef int n = vecs.shape[0]
    cdef int d = vecs.shape[1]
    cdef float* min_dist = <float*>malloc(n * sizeof(float))
    cdef float* curr_dist = <float*>malloc(n * sizeof(float))
    cdef float* centroids = <float*>malloc(K * d * sizeof(float))
    cdef int* belonging = <int*> malloc(n * sizeof(int))
    cdef int* counter = <int*> malloc(K * sizeof(int))

    cdef int[:] init_centroids = np.random.choice(n, K, replace=False).astype(np.int32)
    for i in range(K):
        for j in range(d):
            centroids[i * d + j] = vecs[init_centroids[i], j]

    for epoch in range(iters):
        memset(belonging, 0, n * sizeof(int))
        memset(counter, 0, K * sizeof(int))
        for i in prange(n, schedule='dynamic', nogil=True):
            min_dist[i] = 1e10
            belonging[i] = 0
            for j in range(K):
                curr_dist[i] = 0
                for k in range(d):
                    curr_dist[i] += (centroids[j * d + k] - vecs[i, k]) * (centroids[j * d + k] - vecs[i, k])
                if curr_dist[i] < min_dist[i]:
                    min_dist[i] = curr_dist[i]
                    belonging[i] = j
            counter[belonging[i]] += 1

        memset(centroids, 0, K * d * sizeof(float))
        for i in prange(n, schedule='dynamic', nogil=True):
            for j in range(d):
                centroids[belonging[i] * d + j] += vecs[i, j]

        for i in range(K):
            if counter[i] > 0:
                for j in range(d):
                    centroids[i * d + j] = centroids[i * d + j] / float(counter[i])
            else:
                j = np.random.choice(n)
                for k in range(d):
                    centroids[i * d + k] = vecs[j, k]

    ret = np.array(<float[:K, :d]>centroids)
    ret2 = np.array(<int[:n]>belonging)

    free(min_dist)
    free(curr_dist)
    free(centroids)
    free(belonging)
    free(counter)

    return ret, ret2