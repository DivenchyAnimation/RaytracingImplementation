#pragma once
#include <cuda_runtime.h>
// HD macro for host device set flags/attrib
#define HD __host__ __device__


struct vec3 {
	float x;
	float y;
	float z;

	HD vec3() : x(0.0f), y(0.0f), z(0.0f) {};
	HD vec3(float scalar) : x(scalar), y(scalar), z(scalar) {};
	HD vec3(float x, float y, float z) : x(x), y(y), z(z) {};
	HD vec3 operator+(const vec3 &rh) { return vec3(this->x + rh.x, this->y + rh.y, this->z + rh.z); };
};

struct vec4 {
	float x;
	float y;
	float z;
	float w;

	vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {};
	vec4(float x, float y, float z, float w) : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {};
};

struct mat4 {
	// Identity matrix
	float **matrix;

	mat4() {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				matrix[i][j] = 0.0f;
			}
			matrix[i][i] = 1.0f;
		}
	}
	mat4(vec4 row0, vec4 row1, vec4 row2, vec4 row3) {
		float **m = (float **)malloc(4 * sizeof(float *));
		// Allocate memory for each row
		for (int i = 0; i < 4; i++) {
			float *row = (float*)calloc(4, sizeof(float));
			m[i] = row;
		}

		// Init from vecs
		m[0][0] = row0.x;
		m[0][1] = row0.y;
		m[0][2] = row0.z;
		m[0][3] = row0.w;

		m[1][0] = row1.x;
		m[1][1] = row1.y;
		m[1][2] = row1.z;
		m[1][3] = row1.w;

		m[2][0] = row2.x;
		m[2][1] = row2.y;
		m[2][2] = row2.z;
		m[2][3] = row2.w;

		m[3][0] = row3.x;
		m[3][1] = row3.y;
		m[3][2] = row3.z;
		m[3][3] = row3.w;

		matrix = m;
	}
};

__device__ inline float GPUdot(vec3 a, vec3 b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

//__device__ inline glm::vec3 GPUnormalize(vec3 vector) {
//	float dot = sqrtf(GPUdot(vector, vector));
//	return vector / dot;
//};

#undef HD
