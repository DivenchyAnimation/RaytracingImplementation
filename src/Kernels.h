#include "pch.cuh"
#include <memory>
#include "GPUShape.h"
#pragma once

HD vec3 KernelCalcLightContribution(const GPULight &light, GPUHit *nearestHit, GPURay ray, const GPUShape **shapes, int nShapes);

HD void KernelGenScenePixels(Image &image, int width, int height, GPUCamera *camPos,
                     GPUShape **shapes, int nShapes, const GPULight* lights, int nLights, char *FILENAME, int SCENE, mat4 E);
