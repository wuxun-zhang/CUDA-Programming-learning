#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
using namespace std;

namespace reduce{

namespace CPU{

template<class T>
T reduce(T* src, int n){
	T sum = src[0];
	T c = (T)0.0;

	// Kahan's algorithm
	for (int i = 1; i < n; i++)
    {
        T y = src[i] - c;
        T t = sum + y;
        // Truncation error
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

} // namespace CPU

namespace arm_aarch64{
#ifdef __aarch64__



#endif
} // namespace arm_aarch64

namespace GPU{

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};	

__device__ void warpReduce(volatile float *sm, int tid){
	if(blockDim.x>64) sm[tid] += sm[tid+32];
	if(blockDim.x>32) sm[tid] += sm[tid+16];
	if(blockDim.x>16) sm[tid] += sm[tid+8];
	if(blockDim.x>8) sm[tid] += sm[tid+4];
	if(blockDim.x>4) sm[tid] += sm[tid+2];
	if(blockDim.x>2) sm[tid] += sm[tid+1];
}

// one block Solution
// !!! Low Accupancy !!!
// n : the length of input array
// blockDim: 256
// gridDim: 1
__global__ void reduction_oneBlock(const float* src, int n){
	const unsigned int tid = threadIdx.x;
	__shared__ float sm[256];
	float accum = 0.0f;
	for(int i=tid; i<n; i+=blockDim.x){
		accum += src[i];
	}
	sm[tid] = accum;
	__syncthreads();

	if(blockDim.x > 256){
		if(tid<128)
			sm[tid] = accum = accum + sm[tid+128];
		__syncthreads();
	}

	if(blockDim.x > 128){
		if(tid<64)
			sm[tid] = accum = accum + sm[tid+64];
		__syncthreads();
	}

	if(tid<32){
		warpReduce(sm, tid);
	}

	if(tid == 0){
		src[blockIdx.x] = sm[tid];
	}
}

// multiple blocks solution
// n : length of input array
// blockDim: 256
// gridDim: (n+blockDim.x-1)/blockDim.x
template <class T>
__global__ void
reduce0(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
        	// bank conflicts may happen!
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// n : length of input array
// blockDim: 256
// gridDim: (n+blockDim.x-1)/blockDim.x
template<class T>
__global__ void
reduce1(const T* g_idata, T* g_odata, unsigned int n){
	int tid = threadIdx.x;
	int id = tid + blockIdx.x*blockDim.x;
	T* sdata = SharedMemory<T>();
	sdata[tid] = (id<n) ? g_idata[id] : (T)0.0;
	__syncthreads();

	for(unsigned int s=1; s<blockDim.x; s*=2){
		int index = 2*s*tid;
		if(index<blockDim.x)
			// seriously bank conflicts
			sdata[tid] += sdata[index+s];
		__syncthreads();
	}

	// write results to global mem
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[tid];
}

// n : length of input array
// blockDim: 256
// gridDim: (n+blockDim.x-1)/blockDim.x
template <class T>
__global__ void
reduce2(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    // no divergence or bank conflicts
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// n : length of input array
// blockDim: 256
// gridDim: (n+2*blockDim.x-1)/(2*blockDim.x)
template<class T>
__global__ void
reduce3(const T* g_idata, T* g_odata, unsigned int n){
	T* sdata = SharedMemory<T>();

	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x*(blockDim.x*2) + tid;
	
	T mySum = (id<n) ? g_idata[id] : (T)0.0;
	// each thread block calculate two adjacent block's input elements
	if(id + blockDim.x < n)
		mySum += g_idata[id+blockDim.x];
	sdata[tid] = mySum;
	__syncthreads();

	// do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
        	// register opertaion
        	// lower latency
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

// warp shuffle
// n : length of input array
// blockDim: 256
// gridDim: (n+2*blockDim.x-1)/(2*blockDim.x)
template<T>
__global__ void
reduce4(const T* g_idata, T* g_odata, unsigned int n){
	T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        mySum += g_idata[i+blockSize];

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        __syncthreads();
    }

    // performe warp shuffle when tid < 32
#if (__CUDA_ARCH__ >= 300){
    	if(blockDim.x >= 64)
    		mySum += sdata[tid+32];
    	for(int offset=warpSize/2; offset>0; offset /= 2){
    		mySum += __shfl_down(mySum, offset);
    	}
	}
#else
	// 1: using volatile
	//warpReduce(sdata, tid);

	// 2: fully unroll reduction within a single warp
    // using register, may result in lower latency
    if ((blockDim.x >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockDim.x >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockDim.x >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockDim.x >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockDim.x >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockDim.x >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();

#endif

	if(tid == 0)
		g_odata[blockIdx.x] = mySum;
}

} // namespace GPU

}