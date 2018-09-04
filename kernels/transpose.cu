#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

namespace Transpose{

// write into sharedMem: without bank conflicts
// read from sharedMem: with 16-way bank conflicts	
__global__ void transpose2D(const int size, const int rows, const int cols, const float* src, const int ld_src, float* dst, const int ld_dst){
	extern __shared__ float block[];
	const int tid=threadIdx.x;
	int tx=tid%size;
	int ty=tid/size;
	const int blockPerRow=(cols+size-1)/size;
	int bx=blockIdx.x%blockPerRow;
	int by=blockIdx.x/blockPerRow;

	int xIndex = bx*size+tx;
	int yIndex = by*size+ty;
	if(xIndex<cols && yIndex<rows){
		int in_Index=yIndex*ld_src+xIndex;
		block[ty*size+tx]=src[in_Index];
	}
	__syncthreads();

	// transpose inside a grid
	xIndex = by*size+tx;
	yIndex = bx*size+ty;
	if(xIndex<rows && yIndex<cols){
		int out_Index=yIndex*ld_dst+xIndex;
		// transpose inside a block
		dst[out_Index] = block[tx*size+ty];
	}
}

// without bank conflict
__global__ void transpose2D_opt1(const int size, const int rows, const int cols, const float* src, const int ld_src, float* dst, const int ld_dst){
	__shared__ float block[16][16+1];
	const int tid=threadIdx.x;
	int tx=tid%size;
	int ty=tid/size;
	const int blockPerRow=(cols+size-1)/size;
	int bx=blockIdx.x%blockPerRow;
	int by=blockIdx.x/blockPerRow;

	int xIndex = bx*size+tx;
	int yIndex = by*size+ty;
	if(xIndex<cols && yIndex<rows){
		int in_Index=yIndex*ld_src+xIndex;
		block[ty][tx]=src[in_Index];
	}
	__syncthreads();

	// transpose inside a grid
	xIndex = by*size+tx;
	yIndex = bx*size+ty;
	if(xIndex<rows && yIndex<cols){
		int out_Index=yIndex*ld_dst+xIndex;
		// transpose inside a block
		dst[out_Index] = block[tx][ty];
	}
}

}
