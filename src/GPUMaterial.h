#pragma once
#include "GPUVecOps.h"

struct GPUMaterial {
  vec3 ka; // ambient
  vec3 kd; // diffusion
  vec3 ks; // specular
  float s;      // specular scale/strength
  float reflectivity = 0.0f;

  // For monte carlo
  vec3 ke; // emissive, 0 vec for non lights

  // Default Material
  GPUMaterial() : ka(vec3(0.1f, 0.1f, 0.1f)), kd(vec3(1.0f, 0.0f, 0.0f)), ks(vec3(1.0f, 1.0f, 0.5f)), s(100.0f) {};
  GPUMaterial(const vec3 &ka, const vec3 &kd, const vec3 &ks, float s);
  GPUMaterial(const vec3 &ka, const vec3 &kd, const vec3 &ks, float s, float reflectivity);
  vec3 getMaterialKA();
  vec3 getMaterialKD();
  vec3 getMaterialKS();
  vec3 getMaterialKE() { return this->ke; };
  float getMaterialS();
  float getMaterialReflectivity() { return reflectivity; }
  void setMaterialKA(vec3 ka) { this->ka = ka; }
  void setMaterialKD(vec3 kd) { this->kd = kd; }
  void setMaterialKS(vec3 ks) { this->ks = ks; }
  void setMaterialKE(vec3 ke) { this->ke = ke; }
  void setMaterialS(float s) { this->s = s; }
  void setMaterialReflectivity(float reflectivity) { this->reflectivity = reflectivity; }
};
