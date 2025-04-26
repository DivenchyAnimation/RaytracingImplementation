#pragma once
#include "GPUVecOps.cuh"

struct GPUMaterial {
  vec3 ka; // ambient
  vec3 kd; // diffusion
  vec3 ks; // specular
  float s;      // specular scale/strength
  float reflectivity = 0.0f;

  // For monte carlo
  vec3 ke; // emissive, 0 vec for non lights

  // Default Material
  HD GPUMaterial() : ka(vec3(0.1f, 0.1f, 0.1f)), kd(vec3(1.0f, 0.0f, 0.0f)), ks(vec3(1.0f, 1.0f, 0.5f)), s(100.0f) {};
  HD GPUMaterial(const vec3 &ka, const vec3 &kd, const vec3 &ks, float s);
  HD GPUMaterial(const vec3 &ka, const vec3 &kd, const vec3 &ks, float s, float reflectivity);
  HD vec3 getMaterialKA();
  HD vec3 getMaterialKD();
  HD vec3 getMaterialKS();
  HD vec3 getMaterialKE() { return this->ke; };
  HD float getMaterialS();
  HD float getMaterialReflectivity() { return reflectivity; }
  HD void setMaterialKA(vec3 ka) { this->ka = ka; }
  HD void setMaterialKD(vec3 kd) { this->kd = kd; }
  HD void setMaterialKS(vec3 ks) { this->ks = ks; }
  HD void setMaterialKE(vec3 ke) { this->ke = ke; }
  HD void setMaterialS(float s) { this->s = s; }
  HD void setMaterialReflectivity(float reflectivity) { this->reflectivity = reflectivity; }
};
