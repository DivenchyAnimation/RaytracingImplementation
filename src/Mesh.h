#pragma once
class Mesh : public Shape {
private:
  std::vector<float> posBuf; // list of vertex positions
  std::vector<float> zBuf;   // list of z-vals for image
  std::vector<float> norBuf; // list of vertex normals
  std::vector<float> texBuf; // list of vertex texture coords
  BoundingSphere meshBoundingSphere;

public:
  // Default
  Mesh() {
    this->posBuf = {};
    this->zBuf = {};
    this->norBuf = {};
    this->texBuf = {};
    this->meshBoundingSphere = BoundingSphere();
  }

  // Parameterized
  Mesh(std::vector<float> &posBuf, std::vector<float> &zBuf, std::vector<float> &norBuf, std::vector<float> &texBuf,
       BoundingSphere &meshBoundingSphere, std::shared_ptr<Material> material)
      : Shape() {
    this->posBuf = posBuf;
    this->zBuf = zBuf;
    this->norBuf = norBuf;
    this->texBuf = texBuf;
    this->meshBoundingSphere = meshBoundingSphere;
    this->material = material;
  }

  ~Mesh() override = default;

  virtual std::shared_ptr<Hit> computeIntersection(const Ray &ray, const glm::mat4 modelMat, const glm::mat4 modelMatInv,
                                                   const std::vector<Light> &lights) override;
};
