#pragma once
#include <glm/glm.hpp>
#include "Material.h"
#include <cuda_runtime.h>
#include <vector>
#include "Ray.h"
#include "Light.h"
#include "GPUHit.h"
#include "GPUVecOps.h"


// Virtual is illegal on the device
enum GPUShapeType : int { SPHERE, ELLIPSOID, CUBE, CYLINDER, PLANE };

union ShapeData {
	struct { float radius; } SPHERE;
	struct { float radius; } ELLIPSOID;
	struct { glm::vec3 normal; } PLANE;
};

struct GPUShape {
	GPUShapeType type = GPUShapeType::SPHERE;               // Default shape type
	glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f); // Center of shape
	glm::vec3 rotation = glm::vec3(0.0f, 1.0f, 0.0f); // Axis of rotation
	glm::vec3 scale = glm::vec3(1.0f);
	float rotationAngle = 0.0f;
	ShapeData data;
	Material material = Material();		// Default material
	bool isReflective = false; // Default is not reflective
};

__device__ GPUHit* computeIntersection(const GPUShape *s, const Ray &ray, const glm::mat4 modelMat, const glm::mat4 modelMatInv,
	const Light *lights);
