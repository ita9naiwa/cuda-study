from cudaext import *
import numpy as np
from timeit import timeit
import sys
from icecream import ic


def generate_data(m, n):
    a = np.ones((m, n), dtype='float32')
    return a

def close(a, b):
    r = np.abs(a - b).sum()
    print(r)
    return r  <= 1e-4

def test():
    m, n = 2 ** 4, 2 ** 6
    k = 127
    A = generate_data(m, k)
    B = generate_data(k, n)

    R1 = cuda_mat_mul(A, B)
    R2 = np.dot(A, B)
    close(R1, R2)
test()