#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "cublas_v2.h"

#include "common.h"

__global__ void add_reg_kernel
(float* YtY, float reg, int d) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < d) {
        YtY[idx * d + idx] += reg;
    }
}

__global__ void als_kernel
(int *indices, int *indptr, float *data,
 int nnz, float *X, float *Y, float *partial_gramian, float *pred,
 int n_users, int n_items, int d, float reg,
 int block_beg, int block_size) {

    extern __shared__ float shared_mem[];
    float *r = shared_mem;
    float *Ap = shared_mem + block_size;
    float *p = shared_mem + block_size * 2;
    int i = threadIdx.x;
    for(int u = blockIdx.x; u < n_users; u += gridDim.x) {
        // if(i == 0)
        //     printf("%d %d %d\n", u, indptr[u], indptr[u + 1]);
        if(indptr[u] == indptr[u + 1]){
            __syncthreads();
            continue;
        }

        float *x = &X[u * d + block_beg];
        // calculate residual r
        // r =  b - Ax ;
        // b = \sum_i C_ui * y_i
        // Ax = YtY + \sum_i (C_ui - 1) y_i (y_i^T x_u)
        // b - Ax = [\sum _i (c_ui - (c_ui - 1) * y_i^T x_u)] y_i - YtY x_u

        // (YtY x_u)_i = dot(YtY's i'th row dot x_u
        r[i] = 0;
        for(int j = 0; j < block_size; ++j) {
            r[i] -= partial_gramian[i * block_size + j] * x[j];
        }
        __syncthreads();
        for(int idx = indptr[u]; idx < indptr[u + 1]; ++idx) {
            int iid = indices[idx];
            float c_ui = data[idx];
            float *y = &Y[iid * d + block_beg];
            float y_tx_u = dot(x, y);

            r[i] += ((1.0 - pred[idx]) * c_ui - (c_ui - 1.0) * y_tx_u) * y[i];
            // r[i] += (c_ui - (c_ui - 1.0) * y_tx_u) * y[i];
        }
        p[i] = r[i];
        __syncthreads();
        float rtr = dot(r, r);
        if(rtr <= 1e-10) {
            break;
        }
        for(int cg_i = 0; cg_i < 1; ++cg_i) {
            // A = YtY + \sum_i (c_ui - 1)y_i * y_i^T
            // Ap = YtY * p + sum_i (c_ui -1) y_i * (y_i^T * p)
            Ap[i] = 0;
            for(int j = 0; j < block_size; ++j) {
                Ap[i] += partial_gramian[i * block_size + j] * p[j];
            }
            for(int idx = indptr[u]; idx < indptr[u + 1]; ++idx) {
                int iid = indices[idx];
                float c_ui = data[idx];
                float *y = &Y[iid * d + block_beg];
                float YiTp = dot(y, p);
                float temp = (c_ui - 1.0) * YiTp;
                Ap[i] += temp * y[i];
            }
            __syncthreads();
            float pAp = dot(p, Ap);
            float alpha = rtr / pAp;
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            __syncthreads();
            float rtr_new = dot(r, r);
            if(rtr_new <= 1e-10)
                break;
            float beta = rtr_new / rtr;
            p[i] = r[i] + beta * p[i];
            rtr = rtr_new;
            __syncthreads();
        }
        for(int idx = indptr[u]; idx < indptr[u + 1]; ++idx) {
            int iid = indices[idx];
            float *y = &Y[iid * d + block_beg];
            pred[idx] += x[i] * y[i];
        }
        __syncthreads();
    }
}

int load_matrix_to_cuda_memory(int **dev_indices,
                               int **dev_indptr,
                               float **dev_data,
                               int *indices,
                               int *indptr,
                               float *data,
                               int nnz, int n_users) {
    int *_indices;
    int *_indptr;
    float *_data;
    CUDA_CHECK(cudaMalloc(&_indices, sizeof(int) * nnz));
    CUDA_CHECK(cudaMalloc(&_indptr, sizeof(int) * (n_users + 1)));
    CUDA_CHECK(cudaMalloc(&_data, sizeof(float) * nnz));
    CUDA_CHECK(cudaMemcpy(_indices, indices, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_indptr, indptr, sizeof(int) * (n_users + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_data, data, sizeof(float) * nnz, cudaMemcpyHostToDevice));

    *dev_indices = _indices;
    *dev_indptr = _indptr;
    *dev_data = _data;
}
int load_factors_to_cuda_memory(float **dev_X, float **dev_Y, float *X, float *Y, int n_users, int n_items, int d) {
    float *_X;
    float *_Y;
    CUDA_CHECK(cudaMalloc(&_X, sizeof(float) * n_users * d));
    CUDA_CHECK(cudaMalloc(&_Y, sizeof(float) * n_items * d));
    CUDA_CHECK(cudaMemcpy(_X, X, sizeof(float) * n_users * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_Y, Y, sizeof(float) * n_items * d, cudaMemcpyHostToDevice));
    *dev_X = _X;
    *dev_Y = _Y;
}

cublasHandle_t global_handle = NULL;


int finalize(float *X, float *Y, float *dev_X, float *dev_Y, int *u_indices, int *u_indptr, float *u_data, int *i_indices, int *i_indptr, float *i_data, int n_users, int n_items, int d) {
    CUDA_CHECK(cudaMemcpy(X, dev_X, sizeof(float) * n_users * d, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Y, dev_Y, sizeof(float) * n_items * d, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dev_Y));
    CUDA_CHECK(cudaFree(dev_X));
    CUDA_CHECK(cudaFree(u_indices));
    CUDA_CHECK(cudaFree(i_indices));
    CUDA_CHECK(cudaFree(u_indptr));
    CUDA_CHECK(cudaFree(i_indptr));
    CUDA_CHECK(cudaFree(u_data));
    CUDA_CHECK(cudaFree(i_data));
    if(global_handle != NULL)
        cublasDestroy(global_handle);
}

void _cuALS_iter2
(int *dev_indices, int *dev_indptr, float *dev_data, int nnz,
 float *dev_X, float *dev_Y,
 int n_users, int n_items,
 int d, float reg) {
    int block_size = 64;
    if(d % block_size) {
        printf("dimension %d should be multiple of 64\n", d);
        return;
    }
    if(global_handle == NULL)
        CUBLAS_CHECK(cublasCreate(&global_handle));

    float *dev_YtY, *dev_partial_gramian;
    float *dev_pred;
    float alpha, beta;
    CUDA_CHECK(cudaMalloc(&dev_YtY, sizeof(float) * d * d));
    CUDA_CHECK(cudaMalloc(&dev_pred, sizeof(float) * nnz));
    CUDA_CHECK(cudaMalloc(&dev_partial_gramian, sizeof(float) * block_size * block_size));
    CUDA_CHECK(cudaMemset(dev_pred, 0, sizeof(float) * nnz));
    alpha = 1.0, beta = 0.;
    CUBLAS_CHECK(cublasSgemm(global_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            d,
            d,
            n_items,
            &alpha, dev_Y, d, dev_Y, d,
            &beta, dev_YtY, d
        ));
    CUDA_CHECK(cudaDeviceSynchronize());
    add_reg_kernel<<<d, d>>>(dev_YtY, reg, d);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    for(int block_beg = 0; block_beg < d; block_beg += block_size) {
        // fill partial gramian matrix;
        for(int i = 0; i < block_size; ++i) {
            CUDA_CHECK(cudaMemcpy(dev_partial_gramian + i * block_size,
                dev_YtY + (block_beg + i) * d + block_beg,
                sizeof(float) * block_size,
                cudaMemcpyDeviceToDevice));
        }
        als_kernel<<<numSMs * 8, block_size, sizeof(float) * block_size * 4>>>(
            dev_indices, dev_indptr, dev_data,
            nnz, dev_X, dev_Y, dev_partial_gramian, dev_pred, n_users, n_items, d, reg,
            block_beg, block_size
        );
    }
    cudaFree(dev_YtY);
    cudaFree(dev_partial_gramian);
}


void _cuALS_iter
(int *indices, int *indptr, float* data, int nnz,
 float *X, float *Y, int n_users, int n_items, int d, float reg) {
    int block_size = 32;
    // if(d % block_size) {
    //     printf("dimension %d should be multiple of 128\n", d);
    //     return;
    // }
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    int *dev_indices, *dev_indptr;
    float *dev_data;

    float *dev_X, *dev_Y;
    float *dev_YtY, *dev_partial_gramian;
    float *dev_pred;
    float alpha, beta;
    // Mem. Alloc
    CUDA_CHECK(cudaMalloc(&dev_indices, sizeof(int) * nnz));
    CUDA_CHECK(cudaMalloc(&dev_indptr, sizeof(int) * (n_users + 1)));
    CUDA_CHECK(cudaMalloc(&dev_data, sizeof(float) * nnz));

    CUDA_CHECK(cudaMalloc(&dev_X, sizeof(float) * n_users * d));
    CUDA_CHECK(cudaMalloc(&dev_Y, sizeof(float) * n_items * d));

    CUDA_CHECK(cudaMalloc(&dev_YtY, sizeof(float) * d * d));
    CUDA_CHECK(cudaMalloc(&dev_pred, sizeof(float) * n_users));
    CUDA_CHECK(cudaMalloc(&dev_partial_gramian, sizeof(float) * block_size * block_size));
    //  Mem. init
    CUDA_CHECK(cudaMemcpy(dev_indices, indices, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_indptr, indptr, sizeof(int) * (n_users + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_data, data, sizeof(float) * nnz, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev_X, X, sizeof(float)* n_users * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_Y, Y, sizeof(float)* n_items * d, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dev_pred, 0, sizeof(float) * n_users));

    alpha = 1.0, beta = 0.;
    CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            d,
            d,
            n_items,
            &alpha, dev_Y, d, dev_Y, d,
            &beta, dev_YtY, d
        ));
    CUDA_CHECK(cudaDeviceSynchronize());
    add_reg_kernel<<<d, d>>>(dev_YtY, reg, d);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    for(int block_beg = 0; block_beg < d; block_beg += block_size) {
        // fill partial gramian matrix;
        for(int i = 0; i < block_size; ++i) {
            CUDA_CHECK(cudaMemcpy(dev_partial_gramian + i * block_size,
                dev_YtY + i * d + block_beg,
                sizeof(float) * block_size,
                cudaMemcpyDeviceToDevice));
            }
        als_kernel<<<numSMs * 8, block_size, sizeof(float) * block_size * 4>>>(
            dev_indices, dev_indptr, dev_data,
            nnz, dev_X, dev_Y, dev_YtY, dev_pred, n_users, n_items, d, reg,
            block_beg, block_size
        );
    }
    CUDA_CHECK(cudaMemcpy(X, dev_X, sizeof(float) * n_users * d, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dev_indices));
    CUDA_CHECK(cudaFree(dev_indptr));
    CUDA_CHECK(cudaFree(dev_data));

    CUDA_CHECK(cudaFree(dev_X));
    CUDA_CHECK(cudaFree(dev_Y));
    CUDA_CHECK(cudaFree(dev_YtY));
    CUDA_CHECK(cudaFree(dev_pred));
    CUDA_CHECK(cudaFree(dev_partial_gramian));

}