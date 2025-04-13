// clang-format off
#include "Camera.h"
// clang-format on

class Material {
private:
  glm::vec3 ka; // ambient
  glm::vec3 kd; // diffusion
  glm::vec3 ks; // specular
  float s;      // specular scale/strength

public:
  // Default Material
  Material() : ka(glm::vec3(0.1f, 0.1f, 0.1f)), kd(glm::vec3(1.0f, 0.0f, 0.0f)), ks(glm::vec3(1.0f, 1.0f, 0.5f)), s(100.0f) {};
  Material(const glm::vec3 &ka, const glm::vec3 &kd, const glm::vec3 &ks, float s);
  glm::vec3 getMaterialKA();
  glm::vec3 getMaterialKD();
  glm::vec3 getMaterialKS();
  void setMaterialKA(glm::vec3 ka) { this->ka = ka; }
  void setMaterialKD(glm::vec3 kd) { this->kd = kd; }
  void setMaterialKS(glm::vec3 ks) { this->ks = ks; }
  float getMaterialS();
};
