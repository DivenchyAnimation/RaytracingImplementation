// src/ Routines.cpp
#include "Routines.h"

using std::vector, std::shared_ptr, std::make_shared, std::string, std::sqrt;

// Helper functions
void makeOrthonormalBasis(const glm::vec3 &N, glm::vec3 &T, glm::vec3 &B) {
  if (std::abs(N.x) > 0.1f)
    T = glm::normalize(glm::cross(N, glm::vec3(0, 1, 0)));
  else
    T = glm::normalize(glm::cross(N, glm::vec3(1, 0, 0)));
  B = glm::cross(N, T);
}

// Help from ChatGPT
glm::vec3 cosineSampleHemisphere(const glm::vec3 &N) {
  float r1 = rand01();
  float r2 = rand01();
  float phi = 2.0f * M_PI * r1;
  float cosTheta = std::sqrt(1.0f - r2);
  float sinTheta = std::sqrt(r2);

  // Spherical to Cartesian in local frame
  glm::vec3 T, B;
  makeOrthonormalBasis(N, T, B);
  glm::vec3 H = sinTheta * std::cos(phi) * T + sinTheta * std::sin(phi) * B + cosTheta * N;
  return glm::normalize(H);
}

glm::mat4 buildMVMat(shared_ptr<Shape> &shape) {
  // Create Model matrix and apply transformations
  glm::mat4 modelMat = glm::translate(glm::mat4(1.0f), shape->getPosition());
  // Apply rotation
  modelMat = glm::rotate(modelMat, glm::radians(shape->getRotationAngle()), shape->getRotationAxis());
  modelMat = glm::scale(modelMat, shape->getScale());

  return modelMat;
}

glm::mat4 buildMVMat(shared_ptr<Shape> &shape, glm::mat4 E) {
  // Create Model matrix and apply transformations
  glm::mat4 modelMat = glm::translate(glm::mat4(1.0f), shape->getPosition());
  modelMat = modelMat * E;

  return modelMat;
}

glm::vec3 calcLightContribution(const Light &light, shared_ptr<Hit> nearestHit, Ray &ray, const vector<shared_ptr<Shape>> &shapes) {
  bool isOccluded = isInShadow(nearestHit, light, shapes);

  // For now, use binary shadowing.
  float shadowFactor = isOccluded ? 0.0f : 1.0f;
  shared_ptr<Shape> shape = nearestHit->collisionShape;

  // 2. Compute diffuse shading.
  glm::vec3 L = glm::normalize(light.pos - nearestHit->x);
  float diff = std::max(glm::dot(nearestHit->n, L), 0.0f);
  glm::vec3 diffuse = light.intensity * shape->getMaterial()->getMaterialKD() * diff;

  // 3. Compute specular shading (Phong model).
  glm::vec3 V = glm::normalize(ray.rayOrigin - nearestHit->x); // View direction.
  glm::vec3 H = glm::normalize(L + V);                         // Halfway vector.
  float spec = pow(std::max(glm::dot(nearestHit->n, H), 0.0f), shape->getMaterial()->getMaterialS());
  glm::vec3 specular = light.intensity * shape->getMaterial()->getMaterialKS() * spec;

  // 4. Return the light's contribution scaled by the shadow factor.
  return shadowFactor * (diffuse + specular);
}

glm::vec3 traceRay(Ray &ray, shared_ptr<Hit> &nearestHit, const vector<Light> &lights, const vector<shared_ptr<Shape>> &shapes,
              int depth) {
  // No more bounces
  if (depth <= 0) {
    return glm::vec3(0.0f);
  }
  // Initalize a dummy tVal
  float nearestToCamT = std::numeric_limits<float>::max();
  glm::vec3 finalColor(0.0f, 0.0f, 0.0f); // black bachground

  // Check intersections on each shape
  for (shared_ptr<Shape> shape : shapes) {
    glm::mat4 modelMat = buildMVMat(shape);
    // Obatain inv so that ray is in object space
    glm::mat4 modelMatInv = glm::inverse(modelMat);

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
    glm::vec3 totalLight = nearestHit->collisionShape->getMaterial()->getMaterialKA();
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
    glm::vec3 I = glm::normalize(ray.rayDirection);
    glm::vec3 N = nearestHit->n;
    glm::vec3 R = I - 2.0f * glm::dot(I, N) * N;
    // offset origin to avoid self‐intersection:
    glm::vec3 origin = nearestHit->x + N * 0.001f;
    Ray reflectRay(origin, R);
    glm::vec3 reflCol = traceRay(reflectRay, nearestHit, lights, shapes, depth - 1);
    // blend: local*(1–kr) + reflection*kr
    finalColor = glm::mix(finalColor, reflCol, kr);
  }

  return finalColor;
}

glm::vec3 traceRay(Ray &ray, shared_ptr<Hit> &nearestHit, const vector<Light> &lights, const vector<shared_ptr<Shape>> &shapes,
              int depth, glm::mat4 E) {
  // No more bounces
  if (depth <= 0) {
    return glm::vec3(0.0f);
  }
  // Initalize a dummy tVal
  float nearestToCamT = std::numeric_limits<float>::max();
  glm::vec3 finalColor(0.0f, 0.0f, 0.0f); // black bachground

  // Check intersections on each shape
  for (shared_ptr<Shape> shape : shapes) {
    glm::mat4 modelMat = buildMVMat(shape, E);
    // Obatain inv so that ray is in object space
    glm::mat4 modelMatInv = glm::inverse(modelMat);

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
    glm::vec3 totalLight = nearestHit->collisionShape->getMaterial()->getMaterialKA();
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
    glm::vec3 I = glm::normalize(ray.rayDirection);
    glm::vec3 N = nearestHit->n;
    glm::vec3 R = I - 2.0f * glm::dot(I, N) * N;
    // offset origin to avoid self‐intersection:
    glm::vec3 origin = nearestHit->x + N * 0.001f;
    Ray reflectRay(origin, R);
    glm::vec3 reflCol = traceRay(reflectRay, nearestHit, lights, shapes, depth - 1);
    // blend: local*(1–kr) + reflection*kr
    finalColor = glm::mix(finalColor, reflCol, kr);
  }

  return finalColor;
}

glm::vec3 pathTrace(Ray &ray, const vector<shared_ptr<Shape>> &shapes, const vector<Light> &lights, int depth) {
  // No more bounces
  if (depth <= 0) {
    return glm::vec3(0.0f);
  }
  // Initalize a dummy tVal
  float nearestToCamT = std::numeric_limits<float>::max();
  shared_ptr<Hit> nearestHit = nullptr;
  glm::vec3 finalColor(0.0f, 0.0f, 0.0f); // black bachground

  // Check intersections on each shape
  for (shared_ptr<Shape> shape : shapes) {
    glm::mat4 modelMat = buildMVMat(shape);
    // Obatain inv so that ray is in object space
    glm::mat4 modelMatInv = glm::inverse(modelMat);

    shared_ptr<Hit> curHit;
    curHit = shape->computeIntersection(ray, modelMat, modelMatInv, lights);
    if (curHit->collision && curHit->t < nearestToCamT) {
      nearestToCamT = curHit->t;
      curHit->collisionShape = shape;
      nearestHit = curHit;
    }
  }

  // If hit exists, do shadow test and compute phong color
  if (nearestHit == nullptr) {
    return finalColor;
  }

  // 2) Emitted radiance (if shape is light source):
  const float ambientI = 0.1f;
  glm::vec3 Ka = nearestHit->collisionShape->getMaterial()->getMaterialKA();
  glm::vec3 Ke = nearestHit->collisionShape->getMaterial()->getMaterialKA();

  glm::vec3 L = Ka * ambientI // ambient on every hit
                + Ke;         // add emission (zero on non‑lights)

  for (auto &light : lights) {
    glm::vec3 toL = light.pos - nearestHit->x;
    float d2 = glm::dot(toL, toL);
    glm::vec3 wi = glm::normalize(toL);
    if (!isInShadow(nearestHit, light, shapes)) {
      // simple Lambert:
      float nDotL = std::max(glm::dot(nearestHit->n, wi), 0.0f);
      L += nearestHit->collisionShape->getMaterial()->getMaterialKD() * light.color * (light.intensity / d2) * nDotL;
    }
  }

  glm::vec3 newDir = cosineSampleHemisphere(nearestHit->n);
  float pdf = glm::dot(nearestHit->n, newDir) / M_PI;
  glm::vec3 brdf = nearestHit->collisionShape->getMaterial()->getMaterialKD() / (float)M_PI;
  float cosTheta = glm::dot(nearestHit->n, newDir);
  glm::vec3 throughput = brdf * cosTheta / pdf;

  // Russian roulette
  float q = std::min(throughput.r + throughput.g + throughput.b, 0.95f);
  if (rand01() > q)
    return L; // terminate, return what we’ve gathered so far
  throughput /= q;

  // recurse
  Ray next(nearestHit->x + nearestHit->n * 1e-4f, newDir);
  glm::vec3 Li = pathTrace(next, shapes, lights, depth - 1);
  return L + throughput * Li;
}

void initMaterials(std::vector<std::shared_ptr<Material>> &materials) {
  shared_ptr<Material> redMaterial =
      make_shared<Material>(glm::vec3(0.1f, 0.1f, 0.1f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.5f), 100.0f);
  shared_ptr<Material> greenMaterial =
      make_shared<Material>(glm::vec3(0.1f, 0.1f, 0.1f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.5f), 100.0f);
  shared_ptr<Material> blueMaterial =
      make_shared<Material>(glm::vec3(0.1f, 0.1f, 0.1f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 0.5f), 100.0f);
  shared_ptr<Material> planeMaterial = make_shared<Material>(glm::vec3(0.1f), glm::vec3(1.0f), glm::vec3(0.0f), 0.0f);
  shared_ptr<Material> mirror = make_shared<Material>(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 100.0f, 1.0f);
  shared_ptr<Material> meshMat = make_shared<Material>(glm::vec3(0.1f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 0.5f), 100.0f, 0.0f);
  materials.push_back(redMaterial);
  materials.push_back(greenMaterial);
  materials.push_back(blueMaterial);
  materials.push_back(planeMaterial);
  materials.push_back(mirror);
  materials.push_back(meshMat);
}

// Help from ChatGPT adapting this for camera space
Ray genRayForPixel(int x, int y, int width, int height, shared_ptr<Camera> &cam) {
  float fov = cam->getFOVY();
  float aspect = float(width) / float(height);
  float tanY = tan(fov * 0.5f);
  float tanX = tanY * aspect;

  // Help from ChatGPT
  // Map pixel (x,y) to norm coords
  float u = (((x + 0.5f) / width) * 2.0f - 1.0f) * tanX;
  float v = (((y + 0.5f) / height) * 2.0f - 1.0f) * tanY;

  glm::vec3 camPos = cam->getPosition();
  glm::vec3 forward = glm::normalize(cam->getTarget() - camPos);
  glm::vec3 right = glm::normalize(glm::cross(forward, cam->getWorldUp()));
  glm::vec3 up = glm::cross(right, forward);

  glm::vec3 rayDirWorld = glm::normalize(u * right + v * up + 1.0f * forward);

  return Ray(camPos, rayDirWorld);
}

// Returns if light is occluded
bool isInShadow(shared_ptr<Hit> nearestHit, const Light &light, const vector<shared_ptr<Shape>> &shapes) {

  // Shadow test for each light
  float epsilon = 0.001f;
  glm::vec3 shadowOrigin = nearestHit->x + nearestHit->n * epsilon;

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
    glm::mat4 modelMat = glm::translate(glm::mat4(1.0f), shape->getPosition());
    modelMat = glm::scale(modelMat, shape->getScale());

    // Obatain inv so that ray is in object space
    glm::mat4 modelMatInv = glm::inverse(modelMat);
    shared_ptr<Hit> shadowHit;
    shadowHit = shape->computeIntersection(shadowRay, modelMat, modelMatInv, vector<Light>{light});
    // If a collision occurs and the distance is less than the light's, then this light is occluded.
    if (shadowHit && shadowHit->collision && (shadowHit->t > 0) && (shadowHit->t < lightDistance)) {
      return true;
    }
  }
  return false;
}

//// END OF HELPERS

void genScenePixels(Image &image, int width, int height, shared_ptr<Camera> &cam, const vector<shared_ptr<Shape>> &shapes,
                    vector<shared_ptr<Hit>> &hits, const vector<Light> &lights, string FILENAME, int SCENE) {
  shared_ptr<Hit> nearestHit = make_shared<Hit>();
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      nearestHit = nullptr;
      Ray ray = genRayForPixel(x, y, width, height, cam);

      int depth;
      if (SCENE == 4) {
        depth = 2;
      } else {
        depth = 5;
      }
      glm::vec3 finalColor = traceRay(ray, nearestHit, lights, shapes, depth);
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

void genScenePixels(Image &image, int width, int height, shared_ptr<Camera> &cam, const vector<shared_ptr<Shape>> &shapes,
                    vector<shared_ptr<Hit>> &hits, const vector<Light> &lights, string FILENAME, int SCENE, glm::mat4 E) {
  shared_ptr<Hit> nearestHit = make_shared<Hit>();
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      nearestHit = nullptr;
      Ray ray = genRayForPixel(x, y, width, height, cam);

      int depth;
      if (SCENE == 4) {
        depth = 2;
      } else {
        depth = 5;
      }
      glm::vec3 finalColor = traceRay(ray, nearestHit, lights, shapes, depth, E);
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

void genScenePixelsMonteCarlo(Image &image, int width, int height, std::shared_ptr<Camera> &camPos,
                              const std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                              const std::vector<Light> &lights, std::string FILENAME, int SCENE, glm::mat4 E) {
  shared_ptr<Hit> nearestHit = make_shared<Hit>();
  int totalPixels = width * height;
  int pixelsDone = 0;
  const int SAMPLES = 1000;
  for (int y = 0; y < height; y++) {
    std::cout << "PROGRESS: " << (float(pixelsDone) / float(totalPixels)) * 100.0f << "%" << std::endl;
    for (int x = 0; x < width; x++) {
      glm::vec3 finalColor(0.0f); // Sum of all light
      nearestHit = nullptr;
      int depth = 5;

      // Ray sampling
      for (int s = 0; s < SAMPLES; s++) {
        Ray ray = genRayForPixel(x, y, width, height, camPos);
        finalColor += pathTrace(ray, shapes, lights, depth);
      }

      // Average then Convert color components from [0,1] to [0,255].
      finalColor = finalColor / float(SAMPLES);
      finalColor.r = clamp(finalColor.r);
      finalColor.g = clamp(finalColor.g);
      finalColor.b = clamp(finalColor.b);

      unsigned char rVal = static_cast<unsigned char>(finalColor.r * 255);
      unsigned char gVal = static_cast<unsigned char>(finalColor.g * 255);
      unsigned char bVal = static_cast<unsigned char>(finalColor.b * 255);

      // Set the pixel color in the image.
      image.setPixel(x, y, rVal, gVal, bVal);
      pixelsDone++;
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
  shared_ptr<Shape> redSphere = make_shared<Sphere>(glm::vec3(-0.5f, -1.0f, 1.0f), 1.0f, 1.0f, 0.0f, materials[0]);
  shared_ptr<Shape> greenSphere = make_shared<Sphere>(glm::vec3(0.5f, -1.0f, -1.0f), 1.0f, 1.0f, 0.0f, materials[1]);
  shared_ptr<Shape> blueSphere = make_shared<Sphere>(glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, 1.0f, 0.0f, materials[2]);
  shapes.push_back(redSphere);
  shapes.push_back(greenSphere);
  shapes.push_back(blueSphere);
  Light worldLight(glm::vec3(-2.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), 1.0f);
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
      make_shared<Ellipsoid>(glm::vec3(0.5f, 0.0f, 0.5f), 1.0f, glm::vec3(0.5f, 0.6f, 0.2f), 0.0f, materials[0]);
  shared_ptr<Shape> greenSphere = make_shared<Sphere>(glm::vec3(-0.5f, 0.0f, -0.5f), 1.0f, 1.0f, 0.0f, materials[1]);
  // Plane
  //Plane(glm::vec3 position, glm::vec3 normal, float scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
  shared_ptr<Shape> plane =
      make_shared<Plane>(glm::vec3(0.0f, -1.0f, 0.0f), glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)), 1.0f, 0.0f, materials[3]);
  shapes.push_back(redEllipsoid);
  shapes.push_back(greenSphere);
  shapes.push_back(plane);
  // Create scene lights
  Light worldLightOne(glm::vec3(1.0f, 2.0f, 2.0f), glm::vec3(1.0f), 0.5f);
  Light worldLightTwo(glm::vec3(-1.0f, 2.0f, -1.0f), glm::vec3(1.0f), 0.5f);
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
  shared_ptr<Shape> redSphere = make_shared<Sphere>(glm::vec3(0.5f, -0.7f, 0.5f), 1.0f, 0.3f, 0.0f, materials[0]);
  shared_ptr<Shape> blueSphere = make_shared<Sphere>(glm::vec3(1.0f, -0.7f, 0.0f), 1.0f, 0.3f, 0.0f, materials[2]);
  shared_ptr<Shape> floor = make_shared<Plane>(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, 0.0f, materials[3]);
  shared_ptr<Shape> wall = make_shared<Plane>(glm::vec3(0.0f, 0.0f, -3.0f), glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, 0.0f, materials[3]);
  // Rotate wall
  wall->setRotationAxis(glm::vec3(1.0f, 0.0f, 0.0f));
  wall->setRotationAngle(90.0f);
  shared_ptr<Shape> reflSphere = make_shared<Sphere>(glm::vec3(-0.5f, 0.0f, -0.5f), 1.0f, 1.0f, 0.0f, materials[4]);
  shared_ptr<Shape> reflSphereTwo = make_shared<Sphere>(glm::vec3(1.5f, 0.0f, -1.5f), 1.0f, 1.0f, 0.0f, materials[4]);
  shapes.push_back(redSphere);
  shapes.push_back(blueSphere);
  shapes.push_back(floor);
  shapes.push_back(wall);
  shapes.push_back(reflSphere);
  shapes.push_back(reflSphereTwo);
  // Make lights and push them to vector
  vector<Light> lights;
  Light worldLight(glm::vec3(-1.0f, 2.0f, 1.0f), glm::vec3(1.0f), 0.5f);
  Light worldLightTwo(glm::vec3(0.5f, -0.5f, 0.0f), glm::vec3(1.0f), 0.5f);
  lights.push_back(worldLight);
  lights.push_back(worldLightTwo);
  genScenePixels(image, width, height, cam, shapes, hits, lights, FILENAME, SCENE);
}

void sceneMesh(int width, int height, std::vector<std::shared_ptr<Material>> materials,
               std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits, std::shared_ptr<Camera> &cam,
               std::string FILENAME, std::vector<float> &posBuf, std::vector<float> &zBuf, std::vector<float> &norBuf,
               std::vector<float> &texBuf) {
  Image image(width, height);
  // meshmat --> materials[5]
  BoundingSphere boundSphere(posBuf);
  shared_ptr<Shape> mesh = make_shared<Mesh>(posBuf, zBuf, norBuf, texBuf, boundSphere, materials[5]);
  shapes.push_back(mesh);
  vector<Light> lights;
  Light worldLight(glm::vec3(-1.0f, 1.0f, 1.0f), glm::vec3(1.0f), 1.0f);
  lights.push_back(worldLight);

  genScenePixels(image, width, height, cam, shapes, hits, lights, FILENAME, 6);
}

// E =
//
//     1.5000         0         0    0.3000
//          0    1.4095   -0.5130   -1.5000
//          0    0.5130    1.4095         0
//          0         0         0    1.0000
//

void sceneMeshTransform(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                        std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                        std::shared_ptr<Camera> &cam, std::string FILENAME, std::vector<float> &posBuf, std::vector<float> &zBuf,
                        std::vector<float> &norBuf, std::vector<float> &texBuf) {
  Image image(width, height);
  glm::mat4 E = glm::mat4(glm::vec4(1.5f, 0.0f, 0.0f, 0.0f), glm::vec4(0.0f, 1.4095f, 0.5130f, 0.0f), glm::vec4(0.0f, -0.5130f, 1.4095f, 0.0f),
                glm::vec4(0.3f, -1.5f, 0.0f, 1.0f));
  BoundingSphere boundSphere(posBuf);
  shared_ptr<Shape> mesh = make_shared<Mesh>(posBuf, zBuf, norBuf, texBuf, boundSphere, materials[5]);
  shapes.push_back(mesh);
  vector<Light> lights;
  Light worldLight(glm::vec3(-1.0f, 1.0f, 1.0f), glm::vec3(1.0f), 1.0f);
  lights.push_back(worldLight);

  genScenePixels(image, width, height, cam, shapes, hits, lights, FILENAME, 7, E);
}

void sceneCameraTransform(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                          std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                          std::shared_ptr<Camera> &cam, std::string FILENAME) {

  Image image(width, height);
  // Sphere(glm::vec3 position, float radius, float scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
  shared_ptr<Shape> redSphere = make_shared<Sphere>(glm::vec3(-0.5f, -1.0f, 1.0f), 1.0f, 1.0f, 0.0f, materials[0]);
  shared_ptr<Shape> greenSphere = make_shared<Sphere>(glm::vec3(0.5f, -1.0f, -1.0f), 1.0f, 1.0f, 0.0f, materials[1]);
  shared_ptr<Shape> blueSphere = make_shared<Sphere>(glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, 1.0f, 0.0f, materials[2]);
  shapes.push_back(redSphere);
  shapes.push_back(greenSphere);
  shapes.push_back(blueSphere);
  Light worldLight(glm::vec3(-2.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), 1.0f);
  vector<Light> lights;
  lights.push_back(worldLight);
  // init position(0.0f, 0.0f, 5.0f)
  cam->translateCamera(glm::vec3(-3.0f, 0.0f, -5.0f));
  cam->setFOV(glm::radians(60.0f));
  cam->setTarget(glm::vec3(0.0f, 0.0f, 0.0f)); // Look at origin

  genScenePixels(image, width, height, cam, shapes, hits, lights, FILENAME, 1);
}

void sceneMonteCarlo(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                     std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                     std::shared_ptr<Camera> &cam, std::string FILENAME) {
  Image image(width, height);
  glm::mat4 E = glm::mat4(glm::vec4(1.5f, 0.0f, 0.0f, 0.0f), glm::vec4(0.0f, 1.4095f, 0.5130f, 0.0f), glm::vec4(0.0f, -0.5130f, 1.4095f, 0.0f),
                glm::vec4(0.3f, -1.5f, 0.0f, 1.0f));
  // Make scene
  shared_ptr<Shape> redSphere = make_shared<Sphere>(glm::vec3(-0.5f, -1.0f, 1.0f), 1.0f, 1.0f, 0.0f, materials[0]);
  redSphere->getMaterial()->setMaterialKE(glm::vec3(0.0f));
  shared_ptr<Shape> reflSphere = make_shared<Sphere>(glm::vec3(-0.5f, 0.0f, -0.5f), 1.0f, 1.0f, 0.0f, materials[4]);
  reflSphere->getMaterial()->setMaterialKE(glm::vec3(0.0f));
  shared_ptr<Shape> floor = make_shared<Plane>(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, 0.0f, materials[3]);
  floor->getMaterial()->setMaterialKE(glm::vec3(0.0f));
  shared_ptr<Shape> backWall = make_shared<Plane>(glm::vec3(0.0f, 0.0f, -3.0f), glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, 0.0f, materials[3]);
  backWall->getMaterial()->setMaterialKE(glm::vec3(0.0f));
  // Rotate wall
  backWall->setRotationAxis(glm::vec3(1.0f, 0.0f, 0.0f));
  backWall->setRotationAngle(90.0f);
  // Make an emmissive light
  shared_ptr<Shape> blueLight = make_shared<Sphere>(glm::vec3(1.5f, 0.5f, 1.0f), 1.0f, 0.8f, 0.0f, materials[2]);
  blueLight->getMaterial()->setMaterialKE(glm::vec3(0.0f, 0.0f, 1.0f)); // Make emmissive
  // Lights
  Light worldLight(glm::vec3(1.5f, 0.5f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f), 1.0f);
  vector<Light> lights;
  lights.push_back(worldLight);
  shapes.push_back(redSphere);
  shapes.push_back(reflSphere);
  shapes.push_back(floor);
  shapes.push_back(backWall);
  shapes.push_back(blueLight);

  genScenePixelsMonteCarlo(image, width, height, cam, shapes, hits, lights, FILENAME, 9, E);
}
