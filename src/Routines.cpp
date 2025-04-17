// src/ Routines.cpp
#include "Routines.h"
#include "pch.h"
#include <memory>

using std::vector, std::shared_ptr, std::make_shared, std::string, std::sqrt, glm::vec3, glm::vec4, glm::mat4;

// Helper function
mat4 buildMVMat(shared_ptr<Shape> &shape) {
  // Create Model matrix and apply transformations
  mat4 modelMat = glm::translate(mat4(1.0f), shape->getPosition());
  // Apply rotation
  modelMat = glm::rotate(modelMat, glm::radians(shape->getRotationAngle()), shape->getRotationAxis());
  modelMat = glm::scale(modelMat, shape->getScale());

  return modelMat;
}
float clamp(float x, float minVal = 0.0f, float maxVal = 1.0f) { return std::max(minVal, std::min(x, maxVal)); }

vec3 calcLightContribution(const Light &light, shared_ptr<Hit> nearestHit, Ray ray, const vector<shared_ptr<Shape>> &shapes) {
  bool isOccluded = isInShadow(nearestHit, light, shapes);

  // For now, use binary shadowing.
  float shadowFactor = isOccluded ? 0.0f : 1.0f;
  shared_ptr<Shape> shape = nearestHit->collisionShape;

  // 2. Compute diffuse shading.
  vec3 L = glm::normalize(light.pos - nearestHit->x);
  float diff = std::max(glm::dot(nearestHit->n, L), 0.0f);
  vec3 diffuse = light.intensity * shape->getMaterial()->getMaterialKD() * diff;

  // 3. Compute specular shading (Phong model).
  vec3 V = glm::normalize(ray.rayOrigin - nearestHit->x); // View direction.
  vec3 H = glm::normalize(L + V);                         // Halfway vector.
  float spec = pow(std::max(glm::dot(nearestHit->n, H), 0.0f), shape->getMaterial()->getMaterialS());
  vec3 specular = light.intensity * shape->getMaterial()->getMaterialKS() * spec;

  // 4. Return the light's contribution scaled by the shadow factor.
  return shadowFactor * (diffuse + specular);
}

vec3 traceRay(Ray &ray, shared_ptr<Hit> &nearestHit, const vector<Light> &lights, const vector<shared_ptr<Shape>> &shapes,
              int depth) {
  // No more bounces
  if (depth <= 0) {
    return vec3(0.0f);
  }
  // Initalize a dummy tVal
  float nearestToCamT = std::numeric_limits<float>::max();
  vec3 finalColor(0.0f, 0.0f, 0.0f); // black bachground

  // Check intersections on each shape
  for (shared_ptr<Shape> shape : shapes) {
    mat4 modelMat = buildMVMat(shape);
    // Obatain inv so that ray is in object space
    mat4 modelMatInv = glm::inverse(modelMat);

    shared_ptr<Hit> curHit;
    curHit = shape->computeIntersection(ray, modelMat, modelMatInv, lights);
    if (curHit->collision && curHit->t < nearestToCamT) {
      nearestToCamT = curHit->t;
      curHit->collisionShape = shape;
      nearestHit = curHit;
    }
  }

  // If hit exists, do shadow test and compute phong color
  if ((nearestHit != nullptr)) {
    // init color to ambient, then add light contributions
    vec3 totalLight = nearestHit->collisionShape->getMaterial()->getMaterialKA();
    for (const Light &light : lights) {
      totalLight += calcLightContribution(light, nearestHit, Ray(ray.rayOrigin, ray.rayDirection), shapes);
    }
    finalColor = totalLight;
  } else {
    return finalColor;
  }

  // Reflection step
  float kr = nearestHit->collisionShape->getMaterial()->getMaterialReflectivity();
  if (kr > 0.0f) {
    // reflect direction about normal:
    vec3 I = glm::normalize(ray.rayDirection);
    vec3 N = nearestHit->n;
    vec3 R = I - 2.0f * glm::dot(I, N) * N;
    // offset origin to avoid self‐intersection:
    vec3 origin = nearestHit->x + N * 0.001f;
    Ray reflectRay(origin, R);
    vec3 reflCol = traceRay(reflectRay, nearestHit, lights, shapes, depth - 1);
    // blend: local*(1–kr) + reflection*kr
    finalColor = mix(finalColor, reflCol, kr);
  }

  return finalColor;
}

void initMaterials(std::vector<std::shared_ptr<Material>> &materials) {
  shared_ptr<Material> redMaterial =
      make_shared<Material>(vec3(0.1f, 0.1f, 0.1f), vec3(1.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
  shared_ptr<Material> greenMaterial =
      make_shared<Material>(vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 1.0f, 0.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
  shared_ptr<Material> blueMaterial =
      make_shared<Material>(vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
  shared_ptr<Material> planeMaterial = make_shared<Material>(vec3(0.1f), vec3(1.0f), vec3(0.0f), 0.0f);
  shared_ptr<Material> mirror = make_shared<Material>(vec3(0.0f), vec3(0.0f), vec3(1.0f), 100.0f, 1.0f);
  materials.push_back(redMaterial);
  materials.push_back(greenMaterial);
  materials.push_back(blueMaterial);
  materials.push_back(planeMaterial);
  materials.push_back(mirror);
}

vec3 genRayForPixel(int x, int y, int width, int height, float fov) {
  float planeDistFromCam = 1.0f;
  float halfWidth = planeDistFromCam * tan((fov / 2.0f));
  float halfHeight = halfWidth;

  // Help from ChatGPT
  // Map pixel (x,y) to norm coords
  float u = ((x + 0.5f) / width) * 2.0f - 1.0f;
  float v = ((y + 0.5f) / height) * 2.0f - 1.0f;
  return glm::normalize(vec3(u * halfWidth, v * halfHeight, -planeDistFromCam));
}

// Returns if light is occluded
bool isInShadow(shared_ptr<Hit> nearestHit, const Light &light, const vector<shared_ptr<Shape>> &shapes) {

  // Shadow test for each light
  float epsilon = 0.001f;
  vec3 shadowOrigin = nearestHit->x + nearestHit->n * epsilon;

  Ray shadowRay(shadowOrigin, glm::normalize(light.pos - shadowOrigin));
  float lightDistance = glm::length(light.pos - shadowOrigin);

  for (shared_ptr<Shape> shape : shapes) {
    if (shape->getType() == ShapeType::PLANE) {
      continue;
    }
    if (nearestHit->collisionShape == shape) {
      continue; // Don't double count shape that caused initial hit
    }
    // Create Model matrix and apply transformations
    mat4 modelMat = glm::translate(mat4(1.0f), shape->getPosition());
    modelMat = glm::scale(modelMat, shape->getScale());

    // Obatain inv so that ray is in object space
    mat4 modelMatInv = glm::inverse(modelMat);
    shared_ptr<Hit> shadowHit;
    shadowHit = shape->computeIntersection(shadowRay, modelMat, modelMatInv, vector<Light>{light});
    // If a collision occurs and the distance is less than the light's, then this light is occluded.
    if (shadowHit && shadowHit->collision && (shadowHit->t > 0) && (shadowHit->t < lightDistance)) {
      return true;
    }
  }
  return false;
}

void genScenePixels(Image &image, int width, int height, shared_ptr<Camera> &camPos, const vector<shared_ptr<Shape>> &shapes,
                    vector<shared_ptr<Hit>> &hits, const vector<Light> &lights, string FILENAME, int SCENE) {
  shared_ptr<Hit> nearestHit = make_shared<Hit>();
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      nearestHit = nullptr;
      vec3 rayDir = genRayForPixel(x, y, width, height, camPos->getFOVY());
      Ray ray(camPos->getPosition(), rayDir);

      int depth;
      if (SCENE == 4) {
        depth = 2;
      } else {
        depth = 5;
      }
      vec3 finalColor = traceRay(ray, nearestHit, lights, shapes, depth);
      // Convert color components from [0,1] to [0,255].

      finalColor.r = clamp(finalColor.r);
      finalColor.g = clamp(finalColor.g);
      finalColor.b = clamp(finalColor.b);

      unsigned char rVal = static_cast<unsigned char>(finalColor.r * 255);
      unsigned char gVal = static_cast<unsigned char>(finalColor.g * 255);
      unsigned char bVal = static_cast<unsigned char>(finalColor.b * 255);

      // Set the pixel color in the image.
      image.setPixel(x, y, rVal, gVal, bVal);
    }
  }

  // Write to image
  image.writeToFile(FILENAME);
  std::cout << "Image written to " << FILENAME << std::endl;
}

void sceneOne(int width, int height, std::vector<std::shared_ptr<Material>> materials,
              std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits, std::shared_ptr<Camera> &cam,
              std::string FILENAME) {
  Image image(width, height);
  // Sphere(glm::vec3 position, float radius, float scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
  shared_ptr<Shape> redSphere = make_shared<Sphere>(vec3(-0.5f, -1.0f, 1.0f), 1.0f, 1.0f, 0.0f, materials[0]);
  shared_ptr<Shape> greenSphere = make_shared<Sphere>(vec3(0.5f, -1.0f, -1.0f), 1.0f, 1.0f, 0.0f, materials[1]);
  shared_ptr<Shape> blueSphere = make_shared<Sphere>(vec3(0.0f, 1.0f, 0.0f), 1.0f, 1.0f, 0.0f, materials[2]);
  shapes.push_back(redSphere);
  shapes.push_back(greenSphere);
  shapes.push_back(blueSphere);
  Light worldLight(vec3(-2.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), 1.0f);
  vector<Light> lights;
  lights.push_back(worldLight);

  genScenePixels(image, width, height, cam, shapes, hits, lights, FILENAME, 1);
}

void sceneThree(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                std::shared_ptr<Camera> &cam, std::string FILENAME) {
  Image image(width, height);
  // Ellipsoid(glm::vec3 position, float radius, glm::vec3 scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
  shared_ptr<Shape> redEllipsoid =
      make_shared<Ellipsoid>(vec3(0.5f, 0.0f, 0.5f), 1.0f, vec3(0.5f, 0.6f, 0.2f), 0.0f, materials[0]);
  shared_ptr<Shape> greenSphere = make_shared<Sphere>(vec3(-0.5f, 0.0f, -0.5f), 1.0f, 1.0f, 0.0f, materials[1]);
  // Plane
  //Plane(glm::vec3 position, glm::vec3 normal, float scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
  shared_ptr<Shape> plane =
      make_shared<Plane>(vec3(0.0f, -1.0f, 0.0f), glm::normalize(vec3(0.0f, 1.0f, 0.0f)), 1.0f, 0.0f, materials[3]);
  shapes.push_back(redEllipsoid);
  shapes.push_back(greenSphere);
  shapes.push_back(plane);
  // Create scene lights
  Light worldLightOne(vec3(1.0f, 2.0f, 2.0f), vec3(1.0f), 0.5f);
  Light worldLightTwo(vec3(-1.0f, 2.0f, -1.0f), vec3(1.0f), 0.5f);
  vector<Light> lights;
  lights.push_back(worldLightOne);
  lights.push_back(worldLightTwo);
  genScenePixels(image, width, height, cam, shapes, hits, lights, FILENAME, 3);
}

void sceneReflections(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                      std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                      std::shared_ptr<Camera> &cam, std::string FILENAME, int SCENE) {

  Image image(width, height);
  // Make shapes and push them to vector
  shared_ptr<Shape> redSphere = make_shared<Sphere>(vec3(0.5f, -0.7f, 0.5f), 1.0f, 0.3f, 0.0f, materials[0]);
  shared_ptr<Shape> blueSphere = make_shared<Sphere>(vec3(1.0f, -0.7f, 0.0f), 1.0f, 0.3f, 0.0f, materials[2]);
  shared_ptr<Shape> floor = make_shared<Plane>(vec3(0.0f, -1.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), 1.0f, 0.0f, materials[3]);
  shared_ptr<Shape> wall = make_shared<Plane>(vec3(0.0f, 0.0f, -3.0f), vec3(0.0f, 1.0f, 0.0f), 1.0f, 0.0f, materials[3]);
  // Rotate wall
  wall->setRotationAxis(vec3(1.0f, 0.0f, 0.0f));
  wall->setRotationAngle(90.0f);
  shared_ptr<Shape> reflSphere = make_shared<Sphere>(vec3(-0.5f, 0.0f, -0.5f), 1.0f, 1.0f, 0.0f, materials[4]);
  shared_ptr<Shape> reflSphereTwo = make_shared<Sphere>(vec3(1.5f, 0.0f, -1.5f), 1.0f, 1.0f, 0.0f, materials[4]);
  shapes.push_back(redSphere);
  shapes.push_back(blueSphere);
  shapes.push_back(floor);
  shapes.push_back(wall);
  shapes.push_back(reflSphere);
  shapes.push_back(reflSphereTwo);
  // Make lights and push them to vector
  vector<Light> lights;
  Light worldLight(vec3(-1.0f, 2.0f, 1.0f), vec3(1.0f), 0.5f);
  Light worldLightTwo(vec3(0.5f, -0.5f, 0.0f), vec3(1.0f), 0.5f);
  lights.push_back(worldLight);
  lights.push_back(worldLightTwo);
  genScenePixels(image, width, height, cam, shapes, hits, lights, FILENAME, SCENE);
}
