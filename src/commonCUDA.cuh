#pragma once

#include <assert.h>
#ifdef __CUDAACC__
	#define HD __host__ __device__
#else
	#define HD
#endif

#define CUDA_PI 3.14159265358979f
