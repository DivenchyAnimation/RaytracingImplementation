#include "pch.h"

enum class ShapeType { SPHERE, ELLIPSOID, CUBE, CYLINDER, PLANE };

class Shape {
private:
  ShapeType type;
  glm::vec3 position; // Center of shape
  glm::vec3 normal;   // For planes
  glm::vec3 rotation;
  glm::vec3 scale;
  float radius;
  float height;
  float width;
  float rotationAngle;
  std::shared_ptr<Material> material;

public:
  // Default Shape
  Shape() {
    this->type = ShapeType::SPHERE;
    this->material = std::make_shared<Material>();
    this->radius = 1.0f;
    this->height = 0.0f;
    this->width = 0.0f;
    this->position = glm::vec3(0.0f, 0.0f, 0.0f); // Place in world origin by default
    this->scale = glm::vec3(1.0f, 1.0f, 1.0f);    // Default scale
  }
  // For easy sphere
  Shape(ShapeType type, glm::vec3 position, float radius, glm::vec3 scale, std::shared_ptr<Material> material) {
    this->type = type;
    this->position = position;
    this->radius = radius;
    this->material = material;
    this->height = 0.0f;
    this->width = 0.0f;
    this->scale = scale;                          // Default scale
    this->rotation = glm::vec3(0.0f, 0.0f, 0.0f); // Default rotation
  }
  // For cubics
  Shape(glm::vec3 position, glm::vec3 scale, float width, float height, std::shared_ptr<Material> material) {
    this->position = position;
    this->radius = 0.0f;
    this->height = height;
    this->width = width;
    this->material = material;
    this->scale = scale;
    this->rotation = glm::vec3(0.0f, 0.0f, 0.0f); // Default rotation
  }

  // Plane constructor
  Shape(glm::vec3 position, glm::vec3 normal, glm::vec3 scale, std::shared_ptr<Material> material) {
    this->type = ShapeType::PLANE;
    this->position = position;
    this->normal = normal;
    this->radius = 0.0f;
    this->material = material;
    this->scale = scale;
    this->rotation = glm::vec3(0.0f, 0.0f, 0.0f); // Default rotation
  };

  ~Shape() {};
  glm::vec3 getPosition() { return this->position; };
  float getRadius() { return this->radius; };
  float getRotationAngle() { return this->rotationAngle; };
  std::shared_ptr<Material> getMaterial() { return this->material; };
  ShapeType getType() { return this->type; };
  glm::vec3 getScale() { return this->scale; };
  glm::vec3 getNormal() { return this->normal; };
  glm::vec3 getRotationAxis() { return this->rotation; };
  void setRotationAxis(glm::vec3 rotVec) { this->rotation = rotVec; };
  void setRotationAngle(float angle) { this->rotationAngle = angle; };
};
