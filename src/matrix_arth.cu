#include <iostream>
#include <math.h>
#include <common.h>


__global__ void cudaSum
(float *g_idata, float *g_jdata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n;
        i += blockDim.x * gridDim.x)
      {
        g_odata[i] = g_jdata[i] + g_idata[i];
      }

}

float vec_sum(float *a, float *b, float *r, int n) {
    int n_bytes_in = n * sizeof(float);
    float *dev_idata = NULL, *dev_jdata = NULL,
          *dev_odata = NULL;

    CUDA_CHECK(cudaMalloc(&dev_idata, n_bytes_in));
    CUDA_CHECK(cudaMalloc(&dev_jdata, n_bytes_in));
    CUDA_CHECK(cudaMalloc(&dev_odata, n_bytes_in));
    CUDA_CHECK(cudaMemcpy(dev_idata, a, n_bytes_in, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_jdata, b, n_bytes_in, cudaMemcpyHostToDevice));
    // cudaSum<<< llrint((n / 1024) + 1.0), 1024 >>>(dev_idata, dev_jdata, dev_odata, n);
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    cudaSum<<< 32 * numSMs, 256 >>>(dev_idata, dev_jdata, dev_odata, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(r, dev_odata, n_bytes_in, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_idata));
    CUDA_CHECK(cudaFree(dev_jdata));
    CUDA_CHECK(cudaFree(dev_odata));
    return 0.0;
}

__global__ void MatAdd
(float *A, float *B, float *R, int m, int n) {
    // int col = blockDim.y * blockIdx.y + threadIdx.y;
    // int row = blockDim.x * blockIdx.x + threadIdx.x;
    // int idx = row * n + col;
    // // printf("(%d, %d)\n", row, col);
    // if ((row < m) && (col < n))
    //     R[idx] = A[idx] + B[idx];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < m * n)
        R[idx] = A[idx] + B[idx];
}

float mat_sum(float *a, float *b, float *r, int m, int n) {
    int n_bytes_in = m * n * sizeof(float);
    float *dev_idata = NULL, *dev_jdata = NULL,
          *dev_odata = NULL;

    CUDA_CHECK(cudaMalloc(&dev_idata, n_bytes_in));
    CUDA_CHECK(cudaMalloc(&dev_jdata, n_bytes_in));
    CUDA_CHECK(cudaMalloc(&dev_odata, n_bytes_in));
    CUDA_CHECK(cudaMemcpy(dev_idata, a, n_bytes_in, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_jdata, b, n_bytes_in, cudaMemcpyHostToDevice));
    // int bs = 16;
    // dim3 block_size(bs, bs, 1);
    // dim3 grid_size(ceili(m / block_size.x), ceili(n / block_size.y), 1);

    // MatAdd<<<grid_size, block_size>>>(dev_idata,
    //                                   dev_jdata,
    //                                   dev_odata, m, n);
    int bs = 256;
    dim3 block_size(bs, 1, 1);
    dim3 grid_size(ceili((m * n) / block_size.x), 1, 1);
    MatAdd<<<grid_size, block_size>>>(dev_idata, dev_jdata, dev_odata, m, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(r, dev_odata, n_bytes_in, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_idata));
    CUDA_CHECK(cudaFree(dev_jdata));
    CUDA_CHECK(cudaFree(dev_odata));
    return 0.0;
}

__global__ void MatMul
(float *A, float *B, float *R, int m, int k, int n) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int start_of_A = row * k;
    int start_of_B = col;
    int idx_of_r = row * n + col;
    if((row >= m) || (col >= n))
        return;
    for(int i=0; i < k; ++i){
        R[idx_of_r] += A[start_of_A + i] * B[start_of_B + i * n];
    }
}

float mat_mul(float *a, float *b, float *r, int m, int k, int n) {
    int size_a = m * k * sizeof(float);
    int size_b = n * k * sizeof(float);
    int size_r = m * n * sizeof(float);
    float *dev_a = NULL, *dev_b = NULL, *dev_r = NULL;

    CUDA_CHECK(cudaMalloc(&dev_a, size_a));
    CUDA_CHECK(cudaMalloc(&dev_b, size_b));
    CUDA_CHECK(cudaMalloc(&dev_r, size_r));
    CUDA_CHECK(cudaMemcpy(dev_a, a, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b, size_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dev_r, 0, size_r));
    int bs = 32;
    dim3 block_size(bs, bs, 1);
    dim3 grid_size(ceili(m / block_size.x), ceili(n / block_size.y), 1);

    MatMul<<<grid_size, block_size>>>(dev_a,
                                      dev_b,
                                      dev_r, m, k, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(r, dev_r, size_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_r));
    return 0.0;
}
