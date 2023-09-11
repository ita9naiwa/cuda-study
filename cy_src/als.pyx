import os

cimport cython
from cython cimport floating, integral
from cython.parallel import parallel, prange

import numpy as np
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset

cimport scipy.linalg.cython_blas as cython_blas
cimport scipy.linalg.cython_lapack as cython_lapack


def get_mat_info(mat):
    return mat.indices, mat.indptr, mat.data

def cyALS(user_item_matrix, d, reg, max_iter, method='cpu_cg', num_threads=0):
    if num_threads == 0:
        num_threads = os.cpu_count()
    n_users, n_items = user_item_matrix.shape
    item_user_matrix = user_item_matrix.transpose().tocsr()
    u_indices, u_indptr, u_data = get_mat_info(user_item_matrix)
    i_indices, i_indptr, i_data = get_mat_info(item_user_matrix)
    u_counts = np.ediff1d(u_indices)
    i_counts = np.ediff1d(i_indices)

    X = np.random.normal(0, 0.01, size=(n_users, d)).astype(np.float32)
    Y = np.random.normal(0, 0.01, size=(n_items, d)).astype(np.float32)
    for ep in range(max_iter):
        if method == 'cpu_ialspp':
            _cyALS_iter_ialspp(u_indices, u_indptr, u_data, u_counts,
                               X, Y, n_users, n_items, d, reg, 128, num_threads)
            _cyALS_iter_ialspp(i_indices, i_indptr, i_data, i_counts,
                               Y, X, n_items, n_users, d, reg, 128, num_threads)
        elif method == 'cpu_cg':
            # user iter
            _cyALS_iter_CG(u_indices, u_indptr, u_data, u_counts,
                           X, Y, n_users, n_items, d, reg, num_threads)
            # item iter
            _cyALS_iter_CG(i_indices, i_indptr, i_data, i_counts,
                           Y, X, n_items, n_users, d, reg, num_threads)
        elif method == 'cpu_naive':
            # user iter
            _cyALS_iter(u_indices, u_indptr, u_data, u_counts, X, Y, n_users, n_items, d, reg,num_threads)
            # item iter
            _cyALS_iter(i_indices, i_indptr, i_data, i_counts, Y, X, n_items, n_users, d, reg, num_threads)
        else:
            _cyALS_iter(u_indices, u_indptr, u_data, u_counts, X, Y, n_users, n_items, d, reg,num_threads)
            # item iter
            _cyALS_iter(i_indices, i_indptr, i_data, i_counts, Y, X, n_items, n_users, d, reg, num_threads)
        #loss = _calculate_loss(u_indptr, u_indices, u_data, X, Y, reg)
        #print(loss)
    return X, Y

@cython.cdivision(True)
@cython.boundscheck(False)
def _cyALS_iter(int[:] indices, int[:]indptr, float[:] data, int[:] counts,
           float[:, :] X, float[:, :] Y,
           int n_users, int n_items, int d, float reg, int num_threads):
    cdef float[:, :] YTY = np.dot(Y.T, Y)
    cdef float[:, :] base_A = YTY + reg * np.eye(d, dtype=np.float32)
    cdef int one = 1
    cdef float zero = 0.0
    cdef float c_ui = 0.0
    cdef float temp = 0.0
    cdef int i, j, idx, u
    cdef int err
    cdef float *A
    cdef float *b
    with nogil, parallel(num_threads=num_threads):
        A = <float*> malloc(d * d * sizeof(float));
        b = <float*> malloc(d * sizeof(float))
        for u in prange(n_users, schedule='dynamic', chunksize=8):
            memcpy(A, &base_A[0, 0], d * d * sizeof(float))
            memset(b, 0, d * sizeof(float))
            for idx in range(indptr[u], indptr[u + 1]):
                c_ui = data[idx]
                i = indices[idx]
                # B
                cython_blas.saxpy(&d, &c_ui, &Y[i, 0], &one, b, &one)
                for j in range(d):
                    temp = (c_ui - 1.0) * Y[i, j]
                    cython_blas.saxpy(&d, &temp, &Y[i, 0], &one, A + d * j, &one)
            err = 0
            cython_lapack.sposv(b"U", &d, &one, A, &d, b, &d, &err)

            if err == 0:
                memcpy(&X[u, 0], b, d * sizeof(float))
            else:
                with gil:
                    raise ValueError("cython_lapack.posv failed (err=%i) on row %i. Try "
                                    "increasing the regularization parameter." % (err, u))
        free(A)
        free(b)


@cython.cdivision(True)
@cython.boundscheck(False)
def _cyALS_iter_CG(int[:] indices, int[:]indptr, float[:] data, int[:] counts,
           float[:, :] X, float[:, :] Y,
           int n_users, int n_items, int d, float reg, int num_threads):
    cdef float[:, :] YTY = np.dot(Y.T, Y)
    cdef float[:, :] base_A = YTY + reg * np.eye(d, dtype=np.float32)
    cdef int one = 1
    cdef float zero = 0.0
    cdef float c_ui = 0.0
    cdef float temp = 0.0
    cdef float mone = -1.0
    cdef float fone = 1.0
    cdef float alpha, beta
    cdef int i, j, idx, u
    cdef float *b
    cdef float *r
    cdef float *p
    cdef float *Ap
    cdef float pAp, rtr, rtr_new
    cdef int cg_i
    with nogil, parallel(num_threads=num_threads):
        b = <float*> malloc(d * sizeof(float))
        r = <float*> malloc(d * sizeof(float))
        p = <float*> malloc(d * sizeof(float))
        Ap = <float*> malloc(d * sizeof(float))
        for u in prange(n_users, schedule='dynamic', chunksize=8):
            if indptr[u] == indptr[u + 1]:
                continue
            # calculating r_0
            # https://github.com/benfred/implicit/blob/main/implicit/cpu/_als.pyx#L188
            cython_blas.ssymv(b'U', &d, &mone, &base_A[0, 0], &d, &X[u, 0], &one, &zero, r, &one)
            for idx in range(indptr[u], indptr[u + 1]):
                c_ui = data[idx]
                i = indices[idx]
                # B
                temp = cython_blas.sdot(&d, &Y[i, 0], &one, &X[u, 0], &one)
                temp = c_ui - (c_ui - 1.0) * temp
                cython_blas.saxpy(&d, &temp, &Y[i, 0], &one, r, &one)

            # https://en.wikipedia.org/wiki/Conjugate_gradient_method
            memcpy(p, r, d * sizeof(float))
            rtr = cython_blas.sdot(&d, r, &one, r, &one)
            if rtr <= 1e-10:
                break
            for cg_i in range(3):
                # calculating Ap, without actually calculating (YTCuY)p
                # it decomposes (YTY)p + (YT(Cu - 1Y)p
                cython_blas.ssymv(b'U', &d, &fone, &base_A[0, 0], &d, p, &one, &zero, Ap, &one)
                for idx in range(indptr[u], indptr[u + 1]):
                    i = indices[idx]
                    c_ui = data[idx]
                    temp = (c_ui - 1.0) * cython_blas.sdot(&d, &Y[i, 0], &one, p, &one)
                    cython_blas.saxpy(&d, &temp, &Y[i, 0], &one, Ap, &one)
                # calculating pAp
                pAp = cython_blas.sdot(&d, p, &one, Ap, &one)
                alpha = rtr / pAp
                cython_blas.saxpy(&d, &alpha, p, &one, &X[u, 0], &one)
                alpha = -alpha
                cython_blas.saxpy(&d, &alpha, Ap, &one, r, &one)
                rtr_new = cython_blas.sdot(&d, r, &one, r, &one)
                if rtr_new <= 1e-10:
                    break
                beta = rtr_new / rtr
                for j in range(d):
                    p[j] = r[j] + beta * p[j]
                rtr = rtr_new
        free(b)
        free(r)
        free(p)
        free(Ap)

@cython.cdivision(True)
@cython.boundscheck(True)
def _cyALS_iter_ialspp(int[:] indices, int[:]indptr, float[:] data, int[:] counts,
           float[:, :] X, float[:, :] Y,
           int n_users, int n_items, int d, float reg, int pi=32, int num_threads=0):
    # https://arxiv.org/abs/2110.14044
    if pi >= d:
        return _cyALS_iter_CG(indices, indptr, data, counts, X, Y, n_users, n_items, d, reg, num_threads)
    if (d % pi) != 0:
        print(d % pi)
        raise ValueError("d should be multiple of pi")

    cdef float[:] pred = np.zeros_like(data).astype(np.float32)
    cdef float[:, :] full_gramian = np.dot(Y.T, Y) + np.eye(d).astype(np.float32)
    cdef float *x
    cdef float *y
    cdef float fone = 1.0, mone = -1.0, zero = 0.0
    cdef float temp, c_ui
    cdef int one = 1
    cdef int u, i, j, beg, idx, cg_i, k
    cdef float *r
    cdef float *p
    cdef float *Ap
    cdef float *gramian
    cdef float rtr, rtr_new, alpha, beta, pAp
    for beg in range(0, d, pi):
        with nogil, parallel(num_threads=num_threads):
            gramian = <float*> malloc(pi * pi * sizeof(float))
            for j in range(pi):
                memcpy(gramian + pi * j, &full_gramian[beg + j, beg], pi * sizeof(float))
            r = <float *>malloc(pi * sizeof(float))
            p = <float *>malloc(pi * sizeof(float))
            Ap = <float *>malloc(pi * sizeof(float))
            for u in prange(n_users, schedule='dynamic', chunksize=8):
                if indptr[u] == indptr[u + 1]:
                    continue
                x = &X[u, beg]
                cython_blas.ssymv(b'U', &pi, &mone, gramian, &pi, x, &one, &zero, r, &one)
                for idx in range(indptr[u], indptr[u + 1]):
                    c_ui = data[idx]
                    i = indices[idx]
                    y = &Y[i, beg]
                    temp = cython_blas.sdot(&pi, y, &one, x, &one)
                    temp = c_ui - (c_ui - 1.0) * temp - pred[idx]
                    cython_blas.saxpy(&pi, &temp, y, &one, r, &one)

                memcpy(p, r, pi * sizeof(float))
                rtr = cython_blas.sdot(&pi, r, &one, r, &one)
                if rtr <= 1e-10:
                    break
                for cg_i in range(3):
                    cython_blas.ssymv(b'U', &pi, &fone, gramian, &pi, p, &one, &zero, Ap, &one)
                    for idx in range(indptr[u], indptr[u + 1]):
                        y = &Y[i, beg]
                        i = indices[idx]
                        c_ui = data[idx]
                        temp = (c_ui - 1.0) * cython_blas.sdot(&pi, y, &one, p, &one)
                        cython_blas.saxpy(&pi, &temp, y, &one, Ap, &one)
                    pAp = cython_blas.sdot(&pi, p, &one, Ap, &one)
                    alpha = rtr / pAp
                    cython_blas.saxpy(&pi, &alpha, p, &one, x, &one)
                    alpha = -alpha
                    cython_blas.saxpy(&pi, &alpha, Ap, &one, r, &one)
                    rtr_new = cython_blas.sdot(&pi, r, &one, r, &one)
                    if rtr_new <= 1e-10:
                        break
                    beta = rtr_new / rtr
                    for k in range(pi):
                        p[k] = r[k] + beta * p[k]
                    rtr = rtr_new

                for idx in range(indptr[u], indptr[u + 1]):
                    i = indices[idx]
                    y = &Y[i, beg]
                    pred[idx] += cython_blas.sdot(&pi, y, &one, x, &one)


@cython.cdivision(True)
@cython.boundscheck(False)
def _calculate_loss(integral[:] indptr, integral[:] indices, float[:] data,
                    float[:, :] X, float[:, :] Y, float regularization,
                    int num_threads=0):
    dtype = np.float64 if float is double else np.float32
    cdef integral users = X.shape[0], items = Y.shape[0], u, i, index
    cdef int one = 1, N = X.shape[1]
    cdef float confidence, temp
    cdef float zero = 0.

    cdef float[:, :] YtY = np.dot(np.transpose(Y), Y)

    cdef float * r

    cdef double loss = 0, total_confidence = 0, item_norm = 0, user_norm = 0

    with nogil, parallel(num_threads=num_threads):
        r = <float *> malloc(sizeof(float) * N)
        try:
            for u in prange(users, schedule='static', chunksize=4):
                # calculates (A.dot(Xu) - 2 * b).dot(Xu), without calculating A
                temp = 1.0
                cython_blas.ssymv(b"U", &N, &temp, &YtY[0, 0], &N, &X[u, 0], &one, &zero, r, &one)

                for index in range(indptr[u], indptr[u + 1]):
                    i = indices[index]
                    confidence = data[index]

                    if confidence > 0:
                        temp = -2 * confidence
                    else:
                        temp = 0
                        confidence = -1 * confidence

                    temp = temp + (confidence - 1) * cython_blas.sdot(&N, &Y[i, 0], &one, &X[u, 0], &one)
                    cython_blas.saxpy(&N, &temp, &Y[i, 0], &one, r, &one)

                    total_confidence += confidence
                    loss += confidence

                loss += cython_blas.sdot(&N, r, &one, &X[u, 0], &one)
                user_norm += cython_blas.sdot(&N, &X[u, 0], &one, &X[u, 0], &one)

            for i in prange(items, schedule='dynamic', chunksize=8):
                item_norm += cython_blas.sdot(&N, &Y[i, 0], &one, &Y[i, 0], &one)

        finally:
            free(r)

    loss += regularization * (item_norm + user_norm)
    return loss