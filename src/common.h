#ifndef CYTHON_CUDA_SAMPLES_COMMON_H
#define CYTHON_CUDA_SAMPLES_COMMON_H
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <iostream>


#define rnd(x) (x * rand() / RAND_MAX)


static void CudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s:%d, ", file, line);
        fprintf(stderr, "code: %d, reason: %s\n", error,
                cudaGetErrorString(error));
        exit( EXIT_FAILURE );
    }
}

static const char* cublasGetErrorString(cublasStatus_t status)
{
    // https://stackoverflow.com/a/13041774
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}


static void BlasCheck(cublasStatus_t error, const char *file, int line) {
    if(error != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Error: %s:%d, ", file, line);
        fprintf(stderr, "code: %d, reason: %s\n", error,
                cublasGetErrorString(error));
        exit( EXIT_FAILURE );
    }
}

#define CUDA_CHECK( err ) (CudaCheck( err, __FILE__, __LINE__ ))
#define CUBLAS_CHECK( err ) (BlasCheck( err, __FILE__, __LINE__ ))
static int ceili(float x) {
    return llrint(x + 1.0);
}

#define WARP_SIZE 32

// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__inline__ __device__
float warp_reduce_sum(float val) {
#if __CUDACC_VER_MAJOR__ >= 9
    // __shfl_down is deprecated with cuda 9+. use newer variants
    unsigned int active = __activemask();
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(active, val, offset);
    }
#else
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
#endif
    return val;
}

__inline__ __device__
float dot(const float * a, const float * b) {
    __syncthreads();
    static __shared__ float shared[32];

    // figure out the warp/ position inside the warp
    int warp =  threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // partially reduce the dot product inside each warp using a shuffle
    float val = a[threadIdx.x] * b[threadIdx.x];
    val = warp_reduce_sum(val);

    // write out the partial reduction to shared memory if appropriate
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    // if we we don't have multiple warps, we're done
    if (blockDim.x <= WARP_SIZE) {
        return shared[0];
    }

    // otherwise reduce again in the first warp
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (warp == 0) {
        val = warp_reduce_sum(val);
        // broadcast back to shared memory
        if (threadIdx.x == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    return shared[0];
}

#endif
