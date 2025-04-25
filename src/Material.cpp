
Material::Material(const glm::vec3 &ka, const glm::vec3 &kd, const glm::vec3 &ks, float s) {
  this->ka = ka;
  this->kd = kd;
  this->ks = ks;
  this->s = s;
}

Material::Material(const glm::vec3 &ka, const glm::vec3 &kd, const glm::vec3 &ks, float s, float reflectivity) {
  this->ka = ka;
  this->kd = kd;
  this->ks = ks;
  this->s = s;
  this->reflectivity = reflectivity;
}

glm::vec3 Material::getMaterialKA() { return this->ka; }
glm::vec3 Material::getMaterialKD() { return this->kd; }
glm::vec3 Material::getMaterialKS() { return this->ks; }
float Material::getMaterialS() { return this->s; }
