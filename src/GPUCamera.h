#pragma once
#ifndef GPUCAMERA_H
#define GPUCAMERA_H

#include "GPUVecOps.cuh"

struct GPUCamera {
	enum { ROTATE = 0, TRANSLATE, SCALE };

	HD GPUCamera();
	HD virtual ~GPUCamera();
	HD void setInitDistance(float z) { translations.z = -GPUabs(z); }
	HD void setAspect(float a) { aspect = a; };
	HD void setRotationFactor(float f) { rfactor = f; };
	HD void setTranslationFactor(float f) { tfactor = f; };
	HD void setScaleFactor(float f) { sfactor = f; };
	HD float getFOVY() { return fovy; };
	HD vec3 getPosition() const { return position; }
	HD void translateCamera(vec3 translation) { this->position += translation; }
	HD void setFOV(float f) { this->fovy = f; }
	HD void setTarget(vec3 target) { this->target = target; }
	HD mat4 getViewMatrix() const { return GPULookAt(position, target, worldUp); }
	HD vec3 getTarget() const { return target; }
	HD vec3 getWorldUp() const { return worldUp; }

	float aspect;
	float fovy;
	float znear;
	float zfar;
	vec2 rotations;
	vec3 translations;
	vec2 mousePrev;
	float rfactor;
	float tfactor;
	float sfactor;

	// New members
	vec3 position;
	vec3 target = vec3(0.0f, 0.0f, 0.0f);  // Default camera will look at origin
	vec3 worldUp = vec3(0.0f, 1.0f, 0.0f); // Up vector
};

#endif
