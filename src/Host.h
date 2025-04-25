#pragma once
#include <vector>
#include <memory>
#include "GPUVecOps.h"
#include "GPUShape.h"
#include "GPULight.h"
#include "GPUMaterial.h"
#include "GPUCamera.cuh"
#include "Kernels.h"

void loadShapesOnDevice(std::vector<std::shared_ptr<Shape>> shapes, GPUShape **device_shapesPtrs);
void loadLightsOnDevice(std::vector<Light> lights, GPULight *device_lights);

void HAsceneOne(int width, int height, std::vector<std::shared_ptr<Material>> materials,
    std::vector<std::shared_ptr<Shape>> &shapes, GPUShape **device_shapesPtrs, GPULight *device_lights, std::shared_ptr<Camera> &cam,
    char *FILENAME);
