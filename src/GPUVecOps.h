#pragma once
#include <cuda_runtime.h>
#include "commonCUDA.cuh"
#include <math.h>

struct vec2 {
	float x;
	float y;

	HD vec2() : x(0.0f), y(0.0f){};
	HD vec2(float scalar) : x(scalar), y(scalar) {};
	HD vec2(float x, float y) : x(x), y(y) {};
	HD vec2 operator+(const vec2 &rh) { return vec2(this->x + rh.x, this->y + rh.y); };
	HD vec2 operator/(const float &rh) { return vec2(this->x / rh, this->y / rh); };
	HD vec2 operator*(const float &rh) { return vec2(this->x * rh, this->y * rh); };
};

struct vec3 {
	float x;
	float y;
	float z;

	HD vec3() : x(0.0f), y(0.0f), z(0.0f) {};
	HD vec3(float scalar) : x(scalar), y(scalar), z(scalar) {};
	HD vec3(float x, float y, float z) : x(x), y(y), z(z) {};
	HD vec3 operator+(const vec3 &rh) { return vec3(this->x + rh.x, this->y + rh.y, this->z + rh.z); };
	HD vec3 operator/(const float &rh) { return vec3(this->x / rh, this->y / rh, this->z / rh); };
	HD vec3 operator*(const float &rh) { return vec3(this->x * rh, this->y * rh, this->z * rh); };
	HD void operator+= (const vec3 & rh) { this->x += rh.x; this->y += rh.y; this->z += rh.z; };
};

struct vec4 {
	float x;
	float y;
	float z;
	float w;

	HD vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {};
	HD vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {};
	HD vec4(vec3 vec, float w) : x(vec.x), y(vec.y), z(vec.z), w(w) {};
	HD vec4 operator+(const vec4 &rh) { return vec4(this->x + rh.x, this->y + rh.y, this->z + rh.z, this->w + rh.w); };
	HD vec4 operator/(const float &rh) { return vec4(this->x / rh, this->y / rh, this->z / rh, this->w / rh); };
	HD vec4 operator*(const float &rh) { return vec4(this->x * rh, this->y * rh, this->z * rh, this->w /rh); };

	HD explicit operator vec3() const {
		return vec3(x / w, y / w, z / w);
	}
};

struct mat3 {
	float matrix[3][3];

	// Identity matrix
	HD mat3() {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				matrix[i][j] = (i == j ? 1.0f : 0.0f);
			}
		}
	}

	// Parameterized matrix
	HD mat3(vec3 row0, vec3 row1, vec3 row2) {
		matrix[0][0] = row0.x;
		matrix[0][1] = row0.y;
		matrix[0][2] = row0.z;

		matrix[1][0] = row1.x;
		matrix[1][1] = row1.y;
		matrix[1][2] = row1.z;

		matrix[2][0] = row2.x;
		matrix[2][1] = row2.y;
		matrix[2][2] = row2.z;
	}

	HD float *operator[](size_t row) { assert(row < 3); return matrix[row]; };
	HD const float *operator[](size_t row) const { assert(row < 3); return matrix[row]; };

	// Multiply two mat3s
	HD mat3 operator*(const mat3 &B) const {
		mat3 R;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				float sum = 0.f;
				for (int k = 0; k < 3; ++k)
					sum += matrix[i][k] * B.matrix[k][j];
				R.matrix[i][j] = sum;
			}
		}
		return R;
	}
};

struct mat4 {
	float matrix[4][4];

	// Identity Matrix
	HD mat4() {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				matrix[i][j] = (i == j ? 1.0f : 0.0f);
			}
		}
	}

	// Parameterized matrix
	HD mat4(vec4 row0, vec4 row1, vec4 row2, vec4 row3) {
		// Init from vecs
		matrix[0][0] = row0.x;
		matrix[0][1] = row0.y;
		matrix[0][2] = row0.z;
		matrix[0][3] = row0.w;

		matrix[1][0] = row1.x;
		matrix[1][1] = row1.y;
		matrix[1][2] = row1.z;
		matrix[1][3] = row1.w;

		matrix[2][0] = row2.x;
		matrix[2][1] = row2.y;
		matrix[2][2] = row2.z;
		matrix[2][3] = row2.w;

		matrix[3][0] = row3.x;
		matrix[3][1] = row3.y;
		matrix[3][2] = row3.z;
		matrix[3][3] = row3.w;
	}

	HD float *operator[](size_t row) { assert(row < 4); return matrix[row]; }
	HD const float *operator[](size_t row) const { assert(row < 4); return matrix[row]; }

	// Addition w/ vec3
	HD mat4 operator+(const vec3 &vec) const {
		mat4 T = *this;
		T.matrix[0][3] += vec.x;
		T.matrix[1][3] += vec.y;
		T.matrix[2][3] += vec.z;
		return T;
	}

	HD mat4& operator+=(const vec3 &vec) {
		matrix[0][3] += vec.x;
		matrix[1][3] += vec.y;
		matrix[1][3] += vec.z;
		return *this;
	}

	// Multiplying two mat4
	HD mat4 operator*(const mat4 &B) const {
		mat4 R;
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				float sum = 0.f;
				for (int k = 0; k < 4; ++k)
					sum += matrix[i][k] * B.matrix[k][j];
				R.matrix[i][j] = sum;
			}
		}
		return R;
	}

	// Mult mat4 with vec4
	HD vec4 operator*(const vec4 &v) const {
		vec4 r;
		r.x = matrix[0][0] * v.x + matrix[0][1] * v.y + matrix[0][2] * v.z + matrix[0][3] * v.w;
		r.y = matrix[1][0] * v.x + matrix[1][1] * v.y + matrix[1][2] * v.z + matrix[1][3] * v.w;
		r.z = matrix[2][0] * v.x + matrix[2][1] * v.y + matrix[2][2] * v.z + matrix[2][3] * v.w;
		r.w = matrix[3][0] * v.x + matrix[3][1] * v.y + matrix[3][2] * v.z + matrix[3][3] * v.w;
		return r;
	}
};


// vec3 operations
HD inline vec3 operator*(float scalar, const vec3 &vec) { return vec3(vec.x * scalar, vec.y * scalar, vec.z * scalar); };
HD inline vec3 operator-(const vec3 &a, const vec3 &b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); };

HD inline vec3 cross(const vec3 &a, const vec3 &b) {
	return vec3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}
// vec4 operations

// mat3 operations

// mat4 operations
// Help from ChatGPT for this
HD inline mat4 GPUtranspose(const mat4 &m) {
	mat4 r;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			r[i][j] = m[j][i];
		}
	}
	return r;
}


// Linear algebra
HD inline float GPUdot(vec3 a, vec3 b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

HD inline float GPUdot(vec4 a, vec4 b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
}

HD inline vec3 GPUnormalize(vec3 vector) {
	float dot = sqrtf(GPUdot(vector, vector));
	return vector / dot;
};

HD inline vec4 GPUnormalize(vec4 vector) {
	float dot = sqrtf(GPUdot(vector, vector));
	return vector / dot;
}

HD inline float GPUlength(const vec3 &v) { return sqrt(GPUdot(v, v)); }
HD inline float GPUlength(const vec4 &v) { return sqrt(GPUdot(v, v)); }

// Help from ChatGPT with next two
HD inline mat4 GPUtranslate(const mat4 &M, const vec3 &t) {
	// First make an identity translation matrix
	mat4 T;          
	T[0][3] = t.x;   
	T[1][3] = t.y;  
	T[2][3] = t.z;     
	return M * T;      
}

HD inline mat4 GPUscale(const mat4 &M, const vec3 &s) {
	mat4 S;            // defaults to identity
	S[0][0] = s.x;     // scale X
	S[1][1] = s.y;     // scale Y
	S[2][2] = s.z;     // scale Z
	// S[3][3] is already 1 from the ctor
	return M * S;      // your mat4×mat4 operator
}

// Provided by ChatGPT
HD inline mat4 GPUinverse(const mat4 &m)
{
	// Shortcut to access m.matrix[r][c]
#define M(r,c)  m.matrix[r][c]

	mat4 inv;

	inv.matrix[0][0] = M(1, 1) * M(2, 2) * M(3, 3)
		- M(1, 1) * M(2, 3) * M(3, 2)
		- M(2, 1) * M(1, 2) * M(3, 3)
		+ M(2, 1) * M(1, 3) * M(3, 2)
		+ M(3, 1) * M(1, 2) * M(2, 3)
		- M(3, 1) * M(1, 3) * M(2, 2);

	inv.matrix[0][1] = -M(0, 1) * M(2, 2) * M(3, 3)
		+ M(0, 1) * M(2, 3) * M(3, 2)
		+ M(2, 1) * M(0, 2) * M(3, 3)
		- M(2, 1) * M(0, 3) * M(3, 2)
		- M(3, 1) * M(0, 2) * M(2, 3)
		+ M(3, 1) * M(0, 3) * M(2, 2);

	inv.matrix[0][2] = M(0, 1) * M(1, 2) * M(3, 3)
		- M(0, 1) * M(1, 3) * M(3, 2)
		- M(1, 1) * M(0, 2) * M(3, 3)
		+ M(1, 1) * M(0, 3) * M(3, 2)
		+ M(3, 1) * M(0, 2) * M(1, 3)
		- M(3, 1) * M(0, 3) * M(1, 2);

	inv.matrix[0][3] = -M(0, 1) * M(1, 2) * M(2, 3)
		+ M(0, 1) * M(1, 3) * M(2, 2)
		+ M(1, 1) * M(0, 2) * M(2, 3)
		- M(1, 1) * M(0, 3) * M(2, 2)
		- M(2, 1) * M(0, 2) * M(1, 3)
		+ M(2, 1) * M(0, 3) * M(1, 2);

	inv.matrix[1][0] = -M(1, 0) * M(2, 2) * M(3, 3)
		+ M(1, 0) * M(2, 3) * M(3, 2)
		+ M(2, 0) * M(1, 2) * M(3, 3)
		- M(2, 0) * M(1, 3) * M(3, 2)
		- M(3, 0) * M(1, 2) * M(2, 3)
		+ M(3, 0) * M(1, 3) * M(2, 2);

	inv.matrix[1][1] = M(0, 0) * M(2, 2) * M(3, 3)
		- M(0, 0) * M(2, 3) * M(3, 2)
		- M(2, 0) * M(0, 2) * M(3, 3)
		+ M(2, 0) * M(0, 3) * M(3, 2)
		+ M(3, 0) * M(0, 2) * M(2, 3)
		- M(3, 0) * M(0, 3) * M(2, 2);

	inv.matrix[1][2] = -M(0, 0) * M(1, 2) * M(3, 3)
		+ M(0, 0) * M(1, 3) * M(3, 2)
		+ M(1, 0) * M(0, 2) * M(3, 3)
		- M(1, 0) * M(0, 3) * M(3, 2)
		- M(3, 0) * M(0, 2) * M(1, 3)
		+ M(3, 0) * M(0, 3) * M(1, 2);

	inv.matrix[1][3] = M(0, 0) * M(1, 2) * M(2, 3)
		- M(0, 0) * M(1, 3) * M(2, 2)
		- M(1, 0) * M(0, 2) * M(2, 3)
		+ M(1, 0) * M(0, 3) * M(2, 2)
		+ M(2, 0) * M(0, 2) * M(1, 3)
		- M(2, 0) * M(0, 3) * M(1, 2);

	inv.matrix[2][0] = M(1, 0) * M(2, 1) * M(3, 3)
		- M(1, 0) * M(2, 3) * M(3, 1)
		- M(2, 0) * M(1, 1) * M(3, 3)
		+ M(2, 0) * M(1, 3) * M(3, 1)
		+ M(3, 0) * M(1, 1) * M(2, 3)
		- M(3, 0) * M(1, 3) * M(2, 1);

	inv.matrix[2][1] = -M(0, 0) * M(2, 1) * M(3, 3)
		+ M(0, 0) * M(2, 3) * M(3, 1)
		+ M(2, 0) * M(0, 1) * M(3, 3)
		- M(2, 0) * M(0, 3) * M(3, 1)
		- M(3, 0) * M(0, 1) * M(2, 3)
		+ M(3, 0) * M(0, 3) * M(2, 1);

	inv.matrix[2][2] = M(0, 0) * M(1, 1) * M(3, 3)
		- M(0, 0) * M(1, 3) * M(3, 1)
		- M(1, 0) * M(0, 1) * M(3, 3)
		+ M(1, 0) * M(0, 3) * M(3, 1)
		+ M(3, 0) * M(0, 1) * M(1, 3)
		- M(3, 0) * M(0, 3) * M(1, 1);

	inv.matrix[2][3] = -M(0, 0) * M(1, 1) * M(2, 3)
		+ M(0, 0) * M(1, 3) * M(2, 1)
		+ M(1, 0) * M(0, 1) * M(2, 3)
		- M(1, 0) * M(0, 3) * M(2, 1)
		- M(2, 0) * M(0, 1) * M(1, 3)
		+ M(2, 0) * M(0, 3) * M(1, 1);

	inv.matrix[3][0] = -M(1, 0) * M(2, 1) * M(3, 2)
		+ M(1, 0) * M(2, 2) * M(3, 1)
		+ M(2, 0) * M(1, 1) * M(3, 2)
		- M(2, 0) * M(1, 2) * M(3, 1)
		- M(3, 0) * M(1, 1) * M(2, 2)
		+ M(3, 0) * M(1, 2) * M(2, 1);

	inv.matrix[3][1] = M(0, 0) * M(2, 1) * M(3, 2)
		- M(0, 0) * M(2, 2) * M(3, 1)
		- M(2, 0) * M(0, 1) * M(3, 2)
		+ M(2, 0) * M(0, 2) * M(3, 1)
		+ M(3, 0) * M(0, 1) * M(2, 2)
		- M(3, 0) * M(0, 2) * M(2, 1);

	inv.matrix[3][2] = -M(0, 0) * M(1, 1) * M(3, 2)
		+ M(0, 0) * M(1, 2) * M(3, 1)
		+ M(1, 0) * M(0, 1) * M(3, 2)
		- M(1, 0) * M(0, 2) * M(3, 1)
		- M(3, 0) * M(0, 1) * M(1, 2)
		+ M(3, 0) * M(0, 2) * M(1, 1);

	inv.matrix[3][3] = M(0, 0) * M(1, 1) * M(2, 2)
		- M(0, 0) * M(1, 2) * M(2, 1)
		- M(1, 0) * M(0, 1) * M(2, 2)
		+ M(1, 0) * M(0, 2) * M(2, 1)
		+ M(2, 0) * M(0, 1) * M(1, 2)
		- M(2, 0) * M(0, 2) * M(1, 1);

	// Compute determinant via first row expansion
	float det =
		M(0, 0) * inv.matrix[0][0] +
		M(0, 1) * inv.matrix[1][0] +
		M(0, 2) * inv.matrix[2][0] +
		M(0, 3) * inv.matrix[3][0];

	if (det == 0.0f) {
		// Non-invertible; return identity
		return mat4();
	}

	// Scale the adjugate by 1/det
	float invDet = 1.0f / det;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			inv.matrix[i][j] *= invDet;
		}
	}

#undef M
	return inv;
}

HD inline mat4 GPULookAt(const vec3 &eye,
	const vec3 &center,
	const vec3 &up)
{
	// 2a) forward vector
	vec3 f = GPUnormalize(center - eye);
	// 2b) right vector
	vec3 s = GPUnormalize(cross(f, up));
	// 2c) true up
	vec3 u = cross(s, f);

	// 3) build row-major view matrix
	mat4 M; // identity
	// first row
	M[0][0] = s.x;  M[0][1] = s.y;  M[0][2] = s.z;  M[0][3] = -GPUdot(s, eye);
	// second row
	M[1][0] = u.x;  M[1][1] = u.y;  M[1][2] = u.z;  M[1][3] = -GPUdot(u, eye);
	// third row
	M[2][0] = -f.x;  M[2][1] = -f.y;  M[2][2] = -f.z;  M[2][3] = GPUdot(f, eye);
	// last row
	M[3][0] = 0.0f;  M[3][1] = 0.0f;  M[3][2] = 0.0f;  M[3][3] = 1.0f;

	return M;
}
// Misc

HD inline float GPUmax(const float a, const float b) { return (a > b ? a : b); };
HD inline float GPUabs(const float a) { return (a <= 0 ? a * -1 : a); };
HD inline float GPUabs(const int a) { return (a <= 0 ? a * -1 : a); };
HD inline float GPUradians(const float angle) { return angle * (CUDA_PI / 180.0f); }