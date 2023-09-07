#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <common.h>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "cublas_v2.h"

int ceili(float x) {
    return llrint(x + 1.0);
}


__global__ void __find_belonging
(float *dev_vecs, float *dev_centroids, float *dev_dists, int n, int d, int K) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for(int j = 0; j < K; ++j) {
        for(int k = 0; k < d; ++k) {
            dev_dists[i * K + j] += (dev_vecs[i * d + k] - dev_centroids[j * d + k]) * (dev_vecs[i * d + k] - dev_centroids[j * d + k]);
        }
    }
}

float cu_kmeans
(float *vecs, int n, int d, int K, int iter,
 float *o_centroids, int *o_belonging) {
    auto rd = std::mt19937{std::random_device{}()};

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return -1;
    }

    float *dev_vecs, *dev_dists, *dev_centroids, *dev_aggr_mat;
    CUDA_CHECK(cudaMalloc(&dev_vecs, n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_dists, n * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_centroids, K * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_aggr_mat, K * n * sizeof(float)));
    int * counter = (int*)malloc(K * sizeof(int));
    float *dists = (float*)malloc(n * K * sizeof(float));
    float* aggr_mat = (float*)malloc(K * n * sizeof(float));

    // randomly choose K out of n;
    std::vector<int> arr;
    std::uniform_int_distribution<int> dist(0, n - 1);
    int idx, l;
    l = 0;
    while(l < K){
        idx = dist(rd);
        if(arr.end() != std::find(arr.begin(), arr.end(), idx))
            continue;
        else {
            arr.push_back(idx);
            l += 1;
        }
    }
    for(int i = 0; i < K; ++i) {
        for(int j = 0; j < d; ++j){
            o_centroids[i * d + j] = vecs[arr[i] * d + j];
        }
    }
    CUDA_CHECK(cudaMemcpy(dev_vecs, vecs, sizeof(float)* n * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_centroids, o_centroids, sizeof(float)* K * d, cudaMemcpyHostToDevice));
    for(int epoch = 0; epoch < iter; ++epoch) {
        CUDA_CHECK(cudaMemset(dev_dists, 0, n * K * sizeof(float)));
        dim3 block_size(256, 1, 1);
        dim3 grid_size(ceili(n / block_size.x), 1, 1);
        __find_belonging<<<grid_size, block_size>>>(dev_vecs, dev_centroids, dev_dists, n, d, K);
        CUDA_CHECK(cudaMemcpy(dists, dev_dists, n * K * sizeof(float), cudaMemcpyDeviceToHost));

        memset(o_belonging, -1, n * sizeof(int));
        memset(counter, 0, K * sizeof(int));
        #pragma omp parallel for schedule(dynamic, 4)
        for(int i = 0; i < n; ++i) {
            float curr_max = 123456789.0;
            for(int k = 0; k < K; ++k) {
                if(dists[i * K + k] <= curr_max) {
                    o_belonging[i] = k;
                    curr_max = dists[i * K + k];
                }
            }
            counter[o_belonging[i]] += 1;
        }

        memset(aggr_mat, 0, K * n * sizeof(float));
        #pragma omp parallel for schedule(dynamic, 4)
        for(int i = 0; i < n;++i) {
            aggr_mat[o_belonging[i] * n + i] = 1.0f / (float)counter[o_belonging[i]];
        }
        for(int k = 0; k < K; ++k) {
            if(counter[k] == 0) {
                int idx = dist(rd);
                aggr_mat[k * n + idx] = 1.0f;
                counter[k] = 1;
            }
        }

        float alpha = 1.0;
        float beta = 0.0;

        CUDA_CHECK(cudaMemcpy(dev_aggr_mat, aggr_mat, K * n * sizeof(float), cudaMemcpyHostToDevice));
        // https://peterwittek.com/cublas-matrix-c-style.html
        // 이거 보면서 고치자;

        //  leading dimension은 일반적으로 row size를 두면 되는 것 같다 -_-?;;
        // aggr_mat.shape = (k, n)
        // dev_vecs.shape = (n, d)
        stat = cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d, K, n,
            &alpha,
            dev_vecs, d,
            dev_aggr_mat, n,
            &beta,
            dev_centroids, d);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("cublasSgemm failed\n");
            return -1;
        }
    }
    CUDA_CHECK(cudaMemcpy(o_centroids, dev_centroids, K * d * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_vecs));
    CUDA_CHECK(cudaFree(dev_dists));
    CUDA_CHECK(cudaFree(dev_centroids));
    CUDA_CHECK(cudaFree(dev_aggr_mat));
    free(dists);
    free(counter);
    return 0.0;
}