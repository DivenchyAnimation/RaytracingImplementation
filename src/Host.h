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

void HAsceneOne(int blocks, int numThreads, unsigned char *d_pixels, int numPixels, int width, int height, GPUMaterial *materials,
    GPUShape **device_shapesPtrs, GPULight *device_lights, GPUCamera *cam, char *FILENAME);
