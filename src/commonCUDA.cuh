#pragma once

#ifdef __CUDAACC__
	#define HD __host__ __device__
#else
	#define HD
#endif
