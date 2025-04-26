#pragma once
#include "pch.cuh"

// Virtual is illegal on the device
enum GPUShapeType : int { SPHERE, ELLIPSOID, CUBE, CYLINDER, PLANE };

union ShapeData {
	HD ShapeData() {};

	struct { float radius; } SPHERE;
	struct { float radius; } ELLIPSOID;
	struct { vec3 normal; } PLANE;
};

struct GPUMaterial;
struct GPUShape {
	GPUShapeType type = GPUShapeType::SPHERE;               // Default shape type
	vec3 position = vec3(0.0f, 0.0f, 0.0f); // Center of shape
	vec3 rotation = vec3(0.0f, 1.0f, 0.0f); // Axis of rotation
	vec3 scale = vec3(1.0f);
	float rotationAngle = 0.0f;
	ShapeData data;
	GPUMaterial material;		// Default material
	bool isReflective = false; // Default is not reflective

	HD GPUShape() {};
	HD GPUShape(vec3 pos, vec3 rot, vec3 scale, float rotAngle, GPUMaterial material, bool isRefl) : position(pos), rotation(rot), scale(scale), rotationAngle(rotAngle), material(material), isReflective(isRefl) {};
};

HD GPUHit computeIntersection(const GPUShape *s, const GPURay &ray, const mat4 &modelMat, const mat4 &modelMatInv);
