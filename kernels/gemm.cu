#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
using namespace std;

namespace GEMM_NOTRANS{

namespace CPU{

void matMul(const int m, const int n, const int k, const float* a, int lda, 
	const float* b, int ldb, float* c, int ldc){
	float sum = 0;
	for(int row=0; row<m; row++){
		for(int col=0; col<n;col++){
			for(int kk=0; kk<k; kk++){
				sum+=a[row*lda+kk] * b[kk*ldb+col];
			}
			c[row*ldc+col]=sum;
		}
	}
}

} // namespace CPU

namespace GPU{

// size: 16
// blockDim: size*size
// gridDim: (h+size-1)/size * (w+size-1)/size
__global__ void matMul_opt1(const int size, const int m, const int n, const int k, const float* a, int lda, 
	const float* b, int ldb, float* c, int ldc){
	const int blockPerRow = (n+size-1)/size;
	const int tid=threadIdx.x;
	int tx=tid%size;
	int ty=tid/size;
	int bx=blockIdx.x%blockPerRow;
	int by=blockIdx.x/blockPerRow;

	int xIndex = bx*size+tx;
	int yIndex = by*size+ty;

	float sum = 0.0f;
	if(xIndex<n && yIndex<m){
		int in_index = yIndex*size+xIndex;
		// 
		for(int kk=0; kk<k; kk++)
			sum += a[xIndex*lda+kk] * b[kk*ldb+yIndex];
		c[in_index] = sum;
	}
}

#define TILE_SIZE (32)
// when TILE_SIZE is 16, bank conflicts must be considered carefully
// the common practice is add the extra one column, like sm_a[16][17]
// blockDim: TILE_SIZE * TILE_SIZE
// gridDim: (m+TILE_SIZE-1)/TILE_SIZE * (n+TILE_SIZE-1)/TILE_SIZE
__global__ void matMul_opt2(const int size, const int m, const int n, const int k, const float* a, int lda, 
	const float* b, int ldb, float* c, int ldc){
	const int blockPerRow = (n+TILE_SIZE-1)/TILE_SIZE;
	const int tid=threadIdx.x;
	int tx=tid%TILE_SIZE;
	int ty=tid/TILE_SIZE;
	int bx=blockIdx.x%blockPerRow;
	int by=blockIdx.x/blockPerRow;

	__shared__ float sm_a[TILE_SIZE][TILE_SIZE];
	__shared__ float sm_b[TILE_SIZE][TILE_SIZE];

	int xIndex = bx*TILE_SIZE+tx;
	int yIndex = by*TILE_SIZE+ty;
	float sum = 0.0f;

	for(int i=0;i<(int)ceil((float)k/TILE_SIZE); i++){
		if(i*TILE_SIZE+tx<k && yIndex<m)
		// make sure that the access to each row of b matrix is coalesced
			sm_a[ty][tx] = a[yIndex*lda+i*TILE_SIZE+tx];
		else
			sm_a[ty][tx] = 0.0f;

		// make sure that the access to each row of b matrix is coalesced
		if(i*TILE_SIZE+ty<k && xIndex<n)
			sm_b[ty][tx] = b[(i*TILE_SIZE+ty)*ldc+xIndex];
		else
			sm_b[ty][tx] = 0.0f;
		__syncthreads();

		#pragma unroll
		for(int j=0;j<TILE_SIZE;j++)
			sum += sm_a[ty][j] * sm_b[j][tx];
		__syncthreads();
	}

	if(yIndex<m && xIndex<n)
			c[yIndex*ldc+xIndex] = sum;

}


__global__ void matMul_opt3_register(const int size, const int m, const int n, const int k, const float* a, int lda, 
	const float* b, int ldb, float* c, int ldc){

}

} // namespace GPU

} // namespace GEMM_NOTRANS