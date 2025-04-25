#pragma once
#include <vector>
#include <memory>
#include "GPUShape.h"
#include "GPULight.h"
#include "GPUMaterial.h"
#include "GPUCamera.h"
#include "Kernels.h"
#include "GPUVecOps.cuh"
#include <cuda_runtime.h>


void initMaterials(GPUMaterial *materials);

