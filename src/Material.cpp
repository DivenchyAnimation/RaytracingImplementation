using glm::vec3, glm::vec4, glm::mat4;

Material::Material(const vec3 &ka, const vec3 &kd, const vec3 &ks, float s) {
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

vec3 Material::getMaterialKA() { return this->ka; }
vec3 Material::getMaterialKD() { return this->kd; }
vec3 Material::getMaterialKS() { return this->ks; }
float Material::getMaterialS() { return this->s; }
