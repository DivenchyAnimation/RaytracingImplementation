#pragma once
#ifndef GPUCAMERA_H
#define GPUCAMERA_H

#include "GPUVecOps.cuh"

struct GPUCamera {
	enum { ROTATE = 0, TRANSLATE, SCALE };

	GPUCamera();
	virtual ~GPUCamera();
	void setInitDistance(float z) { translations.z = -GPUabs(z); }
	void setAspect(float a) { aspect = a; };
	void setRotationFactor(float f) { rfactor = f; };
	void setTranslationFactor(float f) { tfactor = f; };
	void setScaleFactor(float f) { sfactor = f; };
	float getFOVY() { return fovy; };
	vec3 getPosition() const { return position; }
	void translateCamera(vec3 translation) { this->position += translation; }
	void setFOV(float f) { this->fovy = f; }
	void setTarget(vec3 target) { this->target = target; }
	mat4 getViewMatrix() const { return GPULookAt(position, target, worldUp); }
	vec3 getTarget() const { return target; }
	vec3 getWorldUp() const { return worldUp; }

	float aspect;
	float fovy;
	float znear;
	float zfar;
	vec2 rotations;
	vec3 translations;
	vec2 mousePrev;
	int state;
	float rfactor;
	float tfactor;
	float sfactor;

	// New members
	vec3 position;
	vec3 target = vec3(0.0f, 0.0f, 0.0f);  // Default camera will look at origin
	vec3 worldUp = vec3(0.0f, 1.0f, 0.0f); // Up vector
};

#endif
