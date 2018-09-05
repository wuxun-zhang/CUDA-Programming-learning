#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
using namespace std;

namespace GEMV{

namespace CPU{

void matrix_vector_multiply(const int m, const int n, const float* a, int lda, 
	const float* b, float* c){
	for(int row=0;row<m;row++){
		float sum = 0.0f;
		for(int col=0;col<n;col++)
			sum+=a[row*lda+col] * b[col];
		c[row] = sum;
	}
} // namespace CPU

namespace GPU{

// size: 16 or 32
// blockDim: size
// gridDim: (m+size-1)/size	
__global__ void mat_vect_multiply_opt1_notrans(const int m, const int n, const float* a, int lda 
	const float* b, float* c){
	unsigned int id = threadIdx.x+blockDim.x*blockIdx.x;
	// avoiding cross-border access
	if(id>=m)
		return;
	// each thread calculates one row of a-matrix
	// non-coalesced access to global memory
	float sum = 0.0f;
	for(int col=0; col<n; col++)
		sum += a[id*lda+col] * b[col];
	c[id] = sum;
}

// size: 32
// blockDim: size
// gridDim: (n+size-1)/size
__global__ void mat_vect_multiply_opt2_trans(const int m, const int n, const float* a, int lda
	const float* b, float* c){
	unsigned int id = threadIdx.x+blockDim.x*blockIdx.x;
	// avoiding cross-border access
	if(id>=n)
		return;
	// each thread calculates one column of a-matrix
	// coalesced access to global memory
	float sum = 0.0f;
	for(int row=0; row<m; row++)
		sum += a[row*lda+id] * b[row];
	c[id] = sum;
}

// size: 32
// blockDim: size
// gridDim: (n+size-1)/size
__global__ void mat_vect_multiply_opt3_trans_constant(const int m, const int n, const float* a, int lda
	const int start, float* c){
	unsigned int id = threadIdx.x+blockDim.x*blockIdx.x;
	// avoiding cross-border access
	if(id>=n)
		return;
	
	// store the b-vector in the constant memory
	// high speed constant memory cache
	float sum = 0.0f;
	int end = start+CONSTANT_SIZE>m ? m:start+CONSTANT_SIZE;
	for(int i=start; i<end; i++)
		sum += a[i*lda+id] * b_constant[i-start];
	c[id] = sum;
}

// size:
// blockDim:
// gridDim:
// Note: sharedSize: the size of one row of transposed a-matrix
// may be larger than blockDim.x  
__global__ void mat_vect_multiply_opt4_trans_sharedMem(const int m, const int n, const float* a, const int lda,
	const float* b, float* c, int sharedSize){
	unsigned int id = threadIdx.x+blockDim.x*blockIdx.x;
	float sum = 0.0f;
	extern __shared__ float sm_b[];
	for(int start=0; start<m; start+=sharedSize){
		__syncthreads();
		#pragma unroll 4
		for(int i=threadIdx.x; i<sharedSize&&i+start<m; i+=blockDim.x){
			sm_b[i] = b[start+i];
		}
		__syncthreads();
		if(n<=id)
			continue;
		int end = start+sharedSize>m?m:start+sharedSize;
		#pragma unroll 8
		for(int i=start; i<end; i++)
			sum += a[i*lda+id] * sm_b[i-start];
		if(id<n)
			c[id] = sum;
	}
}

// size: 16 or 32
// blockDim: size*size
// gridDim: m
// sharedMem: size*size 
__device__ void warpReduce(volatile float* sm, int tid){
	sm[tid] += sm[tid+32];
	sm[tid] += sm[tid+16];
	sm[tid] += sm[tid+8];
	sm[tid] += sm[tid+4];
	sm[tid] += sm[tid+2];
	sm[tid] += sm[tid+1];
} 
// one block calculate one row of a-matrix
// or multiple blocks calculate one row of a-matrix 
__global__ void mat_vect_multiply_opt5_notrans_block(const int m, const int n, const float* a, const int lda,
	const float* b, float* c){
	unsigned int tid = threadIdx.x;
	unsigned int id = tid+blockDim.x*blockIdx.x;
	const int blockSize = blockDim.x;

	extern __shared__ float sm[];
	float accum=0.0f;
	// coalesecd access
	for(int i=tid; i<n; i+=blockDim.x){
		accum+=a[blockIdx.x*lda+i] * b[i];
	}
	sm[tid] = accum;
	__syncthreads();

	// preform reduction
	// completely unroll 
	if(blockSize>512){
		if(tid<256)
			sm[tid] += sm[tid+256];
		__syncthreads();
	}
	if(blockSize>256){
		if(tid<128)
			sm[tid] += sm[tid+128];
		__syncthreads();
	}
	if(blockSize>128){
		if(tid<64)
			sm[tid] += sm[tid+64];
		__syncthreads();
	}

	// when tid is ledd than 32, using warp-synchronization
	if(tid<32)
		warpReduce(sm, tid);
	// write back to global memory
	if(tid==0)
		c[blockIdx.x] = sm[tid];
}

// using a warp to calculate the multiply-add result of one row of a-matrix and b-vector
__global__ void mat_vect_multiply_opt6_notrans_warp(const int m, const int n, const float* a, const int lda,
	const float* b, float* c){
	unsigned int tid = threadIdx.x;
	unsigned int id = tid+blockDim.x*blockIdx.x;
	const int blockSize = blockDim.x;

	extern __shared__ float sm[];
	float accum=0.0f;
	// coalesecd access
	for(int i=tid; i<n; i+=blockDim.x){
		accum+=a[blockIdx.x*lda+i] * b[i];
	}
	sm[tid] = accum;
	__syncthreads();

	// TODO
	
}

} // namespace GPU

}

} // namespace GEMV_NOTRANS