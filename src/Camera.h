#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include <memory>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

class MatrixStack;

class Camera {
public:
  enum { ROTATE = 0, TRANSLATE, SCALE };

  Camera();
  virtual ~Camera();
  void setInitDistance(float z) { translations.z = -std::abs(z); }
  void setAspect(float a) { aspect = a; };
  void setRotationFactor(float f) { rfactor = f; };
  void setTranslationFactor(float f) { tfactor = f; };
  void setScaleFactor(float f) { sfactor = f; };
  float getFOVY() { return fovy; };
  void applyProjectionMatrix(std::shared_ptr<MatrixStack> P) const;
  void applyViewMatrix(std::shared_ptr<MatrixStack> MV) const;
  glm::vec3 getPosition() const { return position; }
  void translateCamera(glm::vec3 translation) { this->position += translation; }
  void setFOV(float f) { this->fovy = f; }
  void setTarget(glm::vec3 target) { this->target = target; }
  glm::mat4 getViewMatrix() const { return glm::lookAt(position, target, worldUp); }
  glm::vec3 getTarget() const { return target; }
  glm::vec3 getWorldUp() const { return worldUp; }

private:
  float aspect;
  float fovy;
  float znear;
  float zfar;
  glm::vec2 rotations;
  glm::vec3 translations;
  glm::vec2 mousePrev;
  int state;
  float rfactor;
  float tfactor;
  float sfactor;

  // New members
  glm::vec3 position;
  glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f);  // Default camera will look at origin
  glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f); // Up vector
};

#endif
