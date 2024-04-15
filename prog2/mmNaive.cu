//#include "matrixMul.h"

__global__ void matrixMulKernel(const float* A, const float* B, float* C, int wA, int wB) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int i = 0; i < wA; ++i) {
        sum += A[row * wA + i] * B[i * wB + col];
    }
    C[row * wB + col] = sum;
}

void natrixNaive(const float* A, const float* B, float* C, int wA, int wB, dim3 threads, dim3 blocks) {

  matrixMulKernel<<<threads, blocks>>>(A, B, C, wA, wB);

}