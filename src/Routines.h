// Helpers now defined in header so it can be used in kernels (CUDA)

void makeOrthonormalBasis(const glm::vec3 &N, glm::vec3 &T, glm::vec3 &B);
float rand01() { return rand() / float(RAND_MAX); }
glm::vec3 cosineSampleHemisphere(const glm::vec3 &N);
glm::mat4 buildMVMat(std::shared_ptr<Shape> &shape);
glm::mat4 buildMVMat(std::shared_ptr<Shape> &shape, glm::mat4 E);
inline float clamp(float x, float minVal = 0.0f, float maxVal = 1.0f) { return std::max(minVal, std::min(x, maxVal)); }

void initMaterials(std::vector<std::shared_ptr<Material>> &materials);
Ray genRayForPixel(int x, int y, int width, int height, std::shared_ptr<Camera> &cam);
bool isInShadow(std::shared_ptr<Hit> nearestHit, const Light &light, const std::vector<std::shared_ptr<Shape>> &shapes);

glm::vec3 calcLightContribution(const Light &light, std::shared_ptr<Hit> nearestHit, Ray &ray,
                                const std::vector<std::shared_ptr<Shape>> &shapes);

//// Core
void genScenePixels(Image &image, int width, int height, std::shared_ptr<Camera> &camPos,
                    const std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                    const std::vector<Light> &lights, std::string FILENAME, int SCENE);
void genScenePixels(Image &image, int width, int height, std::shared_ptr<Camera> &camPos,
                    const std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                    const std::vector<Light> &lights, std::string FILENAME, int SCENE, glm::mat4 E);
void genScenePixelsMonteCarlo(Image &image, int width, int height, std::shared_ptr<Camera> &camPos,
                              const std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                              const std::vector<Light> &lights, std::string FILENAME, int SCENE, glm::mat4 E);

///// SCENES
void sceneOne(int width, int height, std::vector<std::shared_ptr<Material>> materials,
              std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits, std::shared_ptr<Camera> &cam,
              std::string FILENAME);
void sceneThree(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                std::shared_ptr<Camera> &cam, std::string FILENAME);
void sceneReflections(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                      std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                      std::shared_ptr<Camera> &cam, std::string FILENAME, int SCENE);

void sceneMesh(int width, int height, std::vector<std::shared_ptr<Material>> materials,
               std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits, std::shared_ptr<Camera> &cam,
               std::string FILENAME, std::vector<float> &posBuf, std::vector<float> &zBuf, std::vector<float> &norBuf,
               std::vector<float> &texBuf);
void sceneMeshTransform(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                        std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                        std::shared_ptr<Camera> &cam, std::string FILENAME, std::vector<float> &posBuf, std::vector<float> &zBuf,
                        std::vector<float> &norBuf, std::vector<float> &texBuf);
void sceneCameraTransform(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                          std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                          std::shared_ptr<Camera> &cam, std::string FILENAME);
void sceneMonteCarlo(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                     std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                     std::shared_ptr<Camera> &cam, std::string FILENAME);
