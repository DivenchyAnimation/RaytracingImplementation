#pragma once

#include <assert.h>
#define HD __host__ __device__

#define CUDA_PI 3.14159265358979f

HD inline float GPU_MAX_FLT() { return 3.402823e37f; }
// Provided by ChatGPT
HD inline float GPUsqrtf(float x) {
    if (x <= 0.0f) return 0.0f;      // handle zero & negative safely
    // Initial guess: you can do x itself or (x+1)/2 for a bit faster converge:
    float r = x * 0.5f + 0.5f;
    r = 0.5f * (r + x / r);
    r = 0.5f * (r + x / r);
    r = 0.5f * (r + x / r);
    r = 0.5f * (r + x / r);
    // (add one more if you need sub-1-ulp accuracy)
    return r;
};
