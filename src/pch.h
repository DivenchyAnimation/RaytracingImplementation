#pragma once

#include "Image.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// GLM
#define GLM_FORCE_RADIANS
#include "glm/matrix.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// clang-format off
#include "raytri.h"
#include "Camera.h"
#include "Material.h"
#include "Ray.h"
#include "BoundingSphere.h"
#include "Hit.h"
#include "Light.h"
#include "Shape.h"
#include "Sphere.h"
#include "Ellipsoid.h"
#include "Plane.h"
#include "Mesh.h"
