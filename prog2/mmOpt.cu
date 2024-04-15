//#include "matrixMul.h"

__global__ void matrixMulKernel(const float* A, const float* B, float* C, int wA, int wB) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of the output matrix C
    float sum = 0.0f;
    
      for (int i = 0; i < wA; i += 6) {
        int aIdx1 = row * wA + i;
        int aIdx2 = aIdx1 + 1;
        int aIdx3 = aIdx2 + 1;
        int aIdx4 = aIdx3 + 1;
        int aIdx5 = aIdx4 + 1;
        int aIdx6 = aIdx5 + 1;
        int bIdx1 = i * wB + col;
        int bIdx2 = bIdx1 + wB;
        int bIdx3 = bIdx2 + wB;
        int bIdx4 = bIdx3 + wB;
        int bIdx5 = bIdx4 + wB;
        int bIdx6 = bIdx5 + wB;
    
        sum += A[aIdx1] * B[bIdx1] +
               A[aIdx2] * B[bIdx2] +
               A[aIdx3] * B[bIdx3] +
               A[aIdx4] * B[bIdx4] +
               A[aIdx5] * B[bIdx5] +
               A[aIdx6] * B[bIdx6];
      }


    // Write the computed sum to the output matrix C
      C[row * wB + col] = sum;
}



void natrixNaive(const float* A, const float* B, float* C, int wA, int wB, dim3 threads, dim3 blocks) {

  matrixMulKernel<<<threads, blocks>>>(A, B, C, wA, wB);

}