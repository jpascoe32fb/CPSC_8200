// matrixMul.h
#ifndef MATRIXMUL_H
#define MATRIXMUL_H

__global__ void matrixMulKernel(const float* A, const float* B, float* C, int wA, int wB);

#endif 
