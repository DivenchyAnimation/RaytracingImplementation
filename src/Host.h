#pragma once
#include <vector>
#include <memory>
#include "Shape.h"
#include "Light.h"
#include "GPUVecOps.h"
#include "GPUShape.h"
#include "GPUMaterial.h"

void loadShapesOnDevice(std::vector<std::shared_ptr<Shape>> shapes, std::vector<Light> lights);
