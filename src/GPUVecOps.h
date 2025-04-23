#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ inline float GPUdot(glm::vec3 a, glm::vec3 b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__device__ inline glm::vec3 GPUnormalize(glm::vec3 vector) {
	float dot = sqrtf(GPUdot(vector, vector));
	return vector / dot;
};
