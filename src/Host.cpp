// Main routines for device to host communication
#include "Host.h"

void initMaterials(GPUMaterial *mats) {
	GPUMaterial redMaterial = GPUMaterial(vec3(0.1f, 0.1f, 0.1f), vec3(1.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
	GPUMaterial greenMaterial = GPUMaterial(vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 1.0f, 0.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
	GPUMaterial blueMaterial = GPUMaterial(vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
	GPUMaterial planeMaterial = GPUMaterial(vec3(0.1f), vec3(1.0f), vec3(0.0f), 0.0f);
	GPUMaterial mirror = GPUMaterial(vec3(0.0f), vec3(0.0f), vec3(1.0f), 100.0f, 1.0f);
	GPUMaterial meshMat = GPUMaterial(vec3(0.1f), vec3(0.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f, 0.0f);
	mats[0] = redMaterial;
	mats[1] = greenMaterial;
	mats[2] = blueMaterial;
	mats[3] = planeMaterial;
	mats[4] = mirror;
	mats[5] = meshMat;
}
