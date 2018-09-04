#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
using namespace std;

namespace GEMV_NOTRANS{

namespace CPU{

void matrix_vector_multiply(const int m, const int n, const float* a, int lda, 
	const float* b, int ldb, float* c, int ldc){
	
}

}

namespace GPU{



}

} // namespace GEMV_NOTRANS