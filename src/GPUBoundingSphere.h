#pragma once
#include "GPUVecOps.cuh"
#include "commonCUDA.cuh"
#include "GPURay.h"
#include <vector>
#include <math.h>

struct GPUBoundingSphere {
public:
  vec3 center;
  float radius;

  // Default
  HD GPUBoundingSphere() : center(0.0f), radius(0.0f) {};
  HD GPUBoundingSphere(std::vector<float> &posBuf);
  HD bool GPUBoundingSphereIntersect(const GPURay &ray, float t_min, float t_max, float &hitT) const;
};
