#include "pch.cuh"
#include <memory>
#include "GPUShape.h"
#pragma once

HD vec3 KernelCalcLightContribution(const GPULight &light, GPUHit &nearestHit, GPURay &ray, GPUShape **shapes, int nShapes, mat4 &E);
__global__ void KernelGenScenePixels(unsigned char *d_pixels, int numPixels, int width, int height, GPUCamera *cam,
                     GPUShape **shapes, int nShapes, GPULight* lights, int nLights, int SCENE, mat4 E);
