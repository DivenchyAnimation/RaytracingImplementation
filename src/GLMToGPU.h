#pragma once

#pragma once
#include <glm/glm.hpp>
#include "GPUVecOps.cuh"

// glm::vec3 -> your vec3
inline __host__ vec3 toGPU(const glm::vec3 &g) {
	return vec3(g.x, g.y, g.z);
}

// glm::vec4 -> your vec4
inline __host__ vec4 toGPU(const glm::vec4 &g) {
	return vec4(g.x, g.y, g.z, g.w);
}

// glm::mat4 (column-major) -> your row-major mat4
inline __host__ mat4 toGPU(const glm::mat4 &m) {
	// glm::mat4 stores columns, so m[col][row]
	vec4 r0(m[0][0], m[1][0], m[2][0], m[3][0]);
	vec4 r1(m[0][1], m[1][1], m[2][1], m[3][1]);
	vec4 r2(m[0][2], m[1][2], m[2][2], m[3][2]);
	vec4 r3(m[0][3], m[1][3], m[2][3], m[3][3]);
	return mat4(r0, r1, r2, r3);
}
