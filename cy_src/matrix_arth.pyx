cimport cython
from cython.parallel cimport prange
import numpy as np

cdef extern from "../src/matrix_arth.h":
    float vec_sum(float *a, float *b, float *r, int n)
    float mat_sum(float *a, float *b, float *r, int m, int n)
    float mat_mul(float *a, float *b, float *r, int m, int k, int n)

def cuda_vec_sum(float[:] a, float[:] b):
    cdef int n = a.shape[0]
    r = np.zeros(n, dtype=np.float32)
    _cuda_vec_sum(a,b,r,n)
    return r

def _cuda_vec_sum(float[:] a, float[:] b, float[:] r, int n):
    return vec_sum(&a[0], &b[0], &r[0], n)

def cuda_mat_sum(float[:, :] A, float[:, :] B):
    cdef int m, n
    m = A.shape[0]
    n = A.shape[1]
    R = np.zeros_like(A, dtype=np.float32)
    _cuda_mat_sum(A, B, R, m, n)
    return R

def _cuda_mat_sum(float[:, :] A, float[:, :] B, float[:, :] R, int m, int n):
    return mat_sum(&A[0, 0], &B[0, 0], &R[0, 0], m, n)

def cuda_mat_mul(float[:, :] A, float[:, :] B):
    cdef int m, k, n
    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[1]
    R = np.zeros(shape=(m, n), dtype=np.float32)
    _cuda_mat_mul(A, B, R, m, k, n)
    return R

def _cuda_mat_mul(float[:, :] A, float[:, :] B, float[:, :] R, int m, int k, int n):
    return mat_mul(&A[0, 0], &B[0, 0], &R[0, 0], m, k, n)
