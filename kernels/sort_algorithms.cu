#include<iostream>
#include<vector>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
using namespace std;

namespace Sort{

namespace CPU{

void selection_sort(unsigned int *data, int left, int right)
{
    for (int i = left ; i <= right ; ++i)
    {
        unsigned min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j <= right ; ++j)
        {
            unsigned val_j = data[j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

static inline void mySwap(unsigned int& a, unsigned int& b){
	// a = a^b;
	// b = a^b;
	// a = a^b;
	unsigned int temp = a;
	a = b;
	b = temp;
}

// !!! Important !!!
// if we choose a value to be pivot randomly,
// first, we should exchange this value with the first or last element in the array
// then the convetional way can be carried out
int PartSort_v1(unsigned int* data, int left, int right){
	unsigned int& pivot = data[left];
	while(left<right){
		// !!!Important!!!
		// thr order of traversal
		while(left<right && data[right]>=pivot)
			right--;

		while(left<right && data[left]<=pivot)
			left++;
		mySwap(data[left], data[right]);
	}
	mySwap(data[left], pivot);
	return left;
}

int PartSort_v2(unsigned int* data, int left, int right){
	unsigned int& pivot = data[left];
	while(left<right){
		while(left<right && data[right]>=pivot)
			right--;
		data[left] = data[right];
		while(left<right && data[left]<=pivot)
			left++;
		data[right] = data[left];
	}
	data[left] = pivot;
	return left;
}

int PartSort_v3_1(unsigned int* data, int left, int right){
	if(left<right){
		unsigned int& pivot = data[right];
		int cur = left;
		int pre = cur-1;
		while(cur<right){
			// from left to right, choose <pivot
			while(data[cur]<pivot && ++pre!=cur)
				mySwap(data[cur], data[pre]);
			++cur;
		}
		mySwap(data[++pre], data[right]);
		return pre;
	}
	return -1;
}

// more important: forward and backward pointer
int PartSort_v3_2(unsigned int* data, int left, int right){
	if (left<right) {
		unsigned int& pivot = data[left];
		int cur = right;
		int pre = right + 1;
		while (cur>left) {
			// !!! Important !!!
			// from right to left, choose >pivot
			while (data[cur]>pivot && --pre != cur)
				mySwap(data[cur], data[pre]);
			--cur;
		}
		mySwap(data[--pre], data[left]);
		return pre;
	}
	return -1;
}

// recursive 
void quickSort_element(unsigned int *data, int left, int right){
	if(left>=right)
		return;
	int index = PartSort_v2(data, left, right);
	quickSort_element(data, left, index-1);
	quickSort_element(data, index+1, right);
}

void quickSort_ptr(unsigned int *data, int right, int left){
	unsigned int *lptr = data+left;
	unsigned int *rptr = data+right;
	unsigned int pivot = data[(left+right)/2];

	while(lptr<rptr){
		unsigned int lval = *lptr;
		unsigned int rval = *rptr;

		while(rval>pivot){
			rptr--;
			rval = *rptr;
		}

		while(lval<pivot){
			lptr++;
			lval = *lptr;
		}

		if(lptr<rptr){
			*lptr++ = rval;
			*rptr-- = lval;
		}
	}

	int nright = rptr - data;
	int nleft = lptr - data;
	if(left < (rptr-data)){
		quickSort_ptr(data, left, nright);
	}
	if(right > (lptr-data))
		quickSort_ptr(data, nleft, right);
}

}// namespace CPU

namespace GPU{



} // namespace GPU

}