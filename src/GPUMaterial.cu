#include "GPUMaterial.h"

GPUMaterial::GPUMaterial(const vec3 &ka, const vec3 &kd, const vec3 &ks, float s) {
	this->ka = ka;
	this->kd = kd;
	this->ks = ks;
	this->s = s;
}

GPUMaterial::GPUMaterial(const vec3 &ka, const vec3 &kd, const vec3 &ks, float s, float reflectivity) {
	this->ka = ka;
	this->kd = kd;
	this->ks = ks;
	this->s = s;
	this->reflectivity = reflectivity;
}

vec3 GPUMaterial::getMaterialKA() { return this->ka; }
vec3 GPUMaterial::getMaterialKD() { return this->kd; }
vec3 GPUMaterial::getMaterialKS() { return this->ks; }
float GPUMaterial::getMaterialS() { return this->s; }
