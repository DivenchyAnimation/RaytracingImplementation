#pragma once
#include <cuda_runtime.h>
#include "pch.h"

void loadShapesOnDevice(std::vector<std::shared_ptr<Shape>> shapes, std::vector<Light> lights);
