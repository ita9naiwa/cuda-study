cimport cython
from cython cimport floating, integral

import numpy as np

cimport scipy.linalg.cython_blas as cython_blas
cimport scipy.linalg.cython_lapack as cython_lapack


def get_mat_info(mat):
    return mat.indices, mat.indptr, mat.data

def cyALS(user_item_matrix, d, reg, max_iter):
    n_users, n_items = user_item_matrix.shape
    item_user_matrix = user_item_matrix.T
    u_indices, u_indptr, u_data = get_mat_info(user_item_matrix)
    i_indices, i_indptr, i_data = get_mat_info(item_user_matrix)
    u_counts = np.ediff1d(u_indices)
    i_counts = np.ediff1d(i_indices)

    X = np.random.normal(0, 0.01, size=(n_users, d)).astype(np.float32)
    Y = np.random.normal(0, 0.01, size=(n_items, d)).astype(np.float32)

    _cyALS(u_indices, u_indptr, u_data, u_counts,
           X, Y, n_users, n_items, d, reg, max_iter)

def _cyALS(int[:] indices, int[:]indptr, float[:] data, int[:] counts,
           float[:, :] X, float[:, :] Y,
           int n_users, int n_items, int d,float reg, int max_iter):
    cdef float[:, :] YTY = np.dot(Y.T, Y)
    cdef float[:, :] base_A = YTY + reg * np.eye(d)
    cdef int one = 1
    cdef float zero = 0.0
    cdef float c_ui = 0.0
    cdef float temp = 0.0
    cdef float[:] B = np.zeros(d, dtype=np.float32)

    for u in range(n_users):
        A = base_A
        for idx in range(indptr[u], indptr[u + 1]):
            c_ui = data[idx]
            i = indices[idx]
            # B
            cython_blas.saxpy(&d, &c_ui, &Y[i, 0], &one, &B[0], &one)
            for j in range(d):
                temp = (c_ui - 1.0) * Y[i, j]
                cython_blas.saxpy(&d, &temp, &Y[i, 0], &one, &A[j, 0], &one)
        cython_lapack.sposv(b"U", &factors, &one, &A, )