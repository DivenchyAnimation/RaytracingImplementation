class Material {
private:
  glm::vec3 ka; // ambient
  glm::vec3 kd; // diffusion
  glm::vec3 ks; // specular
  float s;      // specular scale/strength
  float reflectivity = 0.0f;

  // For monte carlo
  glm::vec3 ke; // emissive, 0 vec for non lights

public:
  // Default Material
  Material() : ka(glm::vec3(0.1f, 0.1f, 0.1f)), kd(glm::vec3(1.0f, 0.0f, 0.0f)), ks(glm::vec3(1.0f, 1.0f, 0.5f)), s(100.0f) {};
  Material(const glm::vec3 &ka, const glm::vec3 &kd, const glm::vec3 &ks, float s);
  Material(const glm::vec3 &ka, const glm::vec3 &kd, const glm::vec3 &ks, float s, float reflectivity);
  glm::vec3 getMaterialKA();
  glm::vec3 getMaterialKD();
  glm::vec3 getMaterialKS();
  glm::vec3 getMaterialKE() { return this->ke; };
  float getMaterialS();
  float getMaterialReflectivity() { return reflectivity; }
  void setMaterialKA(glm::vec3 ka) { this->ka = ka; }
  void setMaterialKD(glm::vec3 kd) { this->kd = kd; }
  void setMaterialKS(glm::vec3 ks) { this->ks = ks; }
  void setMaterialKE(glm::vec3 ke) { this->ke = ke; }
  void setMaterialS(float s) { this->s = s; }
  void setMaterialReflectivity(float reflectivity) { this->reflectivity = reflectivity; }
};
