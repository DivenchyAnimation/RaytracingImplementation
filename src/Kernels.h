#include <cuda_runtime.h>
#include "GPUShape.h"
#include "GPUHit.h"
#include "GPULight.h"
#include "GPURay.h"
#include "GPUVecOps.h"
#pragma once

HD vec3 KernelcalcLightContribution(const GPULight &light, GPUHit *nearestHit, GPURay ray, const GPUShape **shapes, int nShapes);
