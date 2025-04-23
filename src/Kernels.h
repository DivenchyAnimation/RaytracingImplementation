#define GLM_FORCE_CUDA
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>  // or whatever you need
#include "GPUShape.h"
#include "GPUHit.h"
#include "GPUVecOps.h"
#pragma once

__device__ glm::vec3 KernerlcalcLightContribution(const Light &light, GPUHit *nearestHit, Ray ray, const GPUShape **shapes, int nShapes);
