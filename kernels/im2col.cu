#include<iostream>
#include<algorithms>
using namespace std;

namespace im2col{

static inline bool is_a_ge_zero_and_a_lt_b(const int a, const int b){
	return (a>=0 && a<b);
}

namespace CPU{

template<class T>
void im2col_nchw(const int channel, const T* data_im, const int height_im, 
	const int width_im, const int kernel_h, const int kernel_w, 
	const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int height_col, const int width_col, T* data_col){
	const int channel_size = width_im * height_im;
	for(int c=0; c<channel; c++){
	  for(int kh=0; kh<kernel_h; kh++){
		for(int kw=0; kw<kernel_w; kw++){
		  int ih = kh-pad_h;
	      for(int oh=0; oh<height_col; oh++){
	      	if(is_a_ge_zero_and_a_lt_b(ih, height_im)){
	      	   int iw = kw-pad_w;
	      	   for(int ow=0; ow<width_col; ow++){
	      	   	  *(data_col++) = is_a_ge_zero_and_a_lt_b(iw, width_im)?data_im[oh*width+ow]:(T)0.0;
	      	   	  iw+=stride_w;
	      	   }
	      	}else{
	      		for(int ow=0; ow<width_col; ow++)
	      			*(data_col++) = (T)0.0;
	      	}
	      	ih+=stride_h;
		  }
	    }
	  }
	  data_im+=channel_size;
	}
}

} // namespace CPU


namespace GPU{

// blockDim: 16*16
// gridDim: (channel*height_col*width_col+blockDim.x-1)/blockDim.x
template<class T>
__global__ void
im2col_nchw(const int n, const T* data_im, const int height_im, 
	const int width_im, const int kernel_h, const int kernel_w, 
	const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int height_col, const int width_col, T* data_col){
	const unsigned int tid = threadIdx.x;
	for(int index=tid+blockDim.x*blockIdx.x; index<n; index+=blockDim.x*gridDim.x){
		// height idx of output feature maps
		const int h_index = index / width_col;
		const int h_col = h_index % height_col;
		const int w_col = index % width_col;
		const int c_im = h_index / height_col;
		const int c_col = c_im*kernel_w*kernel_h;
		const h_offset = h_col*stride_h-pad_h;
		const w_offset = w_col*stride_w-pad_w;

		T* data_col_ptr = data_col + 
			(c_col*height_col+h_col)*width_col+w_col;
		const T* data_im_ptr = data_im +
			(c_im*height_im+h_offset)*width_im+w_offset;

		for(int i=0; i<kernel_h; i++){
			for(int j=0; j<kernel_w; j++){
				int h_im = h_offset+i;
				int w_im = w_offset+j;
				*data_col_ptr = (is_a_ge_zero_and_a_lt_b(h_im, height_im) &&
					is_a_ge_zero_and_a_lt_b(w_im, width_im)) ? data_im_ptr[i*width_im+j]:(T)0.0;
				data_col_ptr += height_col*width_col;
			}
		}		
	}
}


} // namespace GPU

} // namespace im2col