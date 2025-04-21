void initMaterials(std::vector<std::shared_ptr<Material>> &materials);
Ray genRayForPixel(int x, int y, int width, int height, std::shared_ptr<Camera> &cam);
bool isInShadow(std::shared_ptr<Hit> nearestHit, const Light &light, const std::vector<std::shared_ptr<Shape>> &shapes);
glm::vec3 calcLightContribution(const Light &light, std::shared_ptr<Hit> nearestHit, Ray &ray,
                                const std::vector<std::shared_ptr<Shape>> &shapes);
void genScenePixels(Image &image, int width, int height, std::shared_ptr<Camera> &camPos,
                    const std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                    const std::vector<Light> &lights, std::string FILENAME, int SCENE);
void genScenePixels(Image &image, int width, int height, std::shared_ptr<Camera> &camPos,
                    const std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                    const std::vector<Light> &lights, std::string FILENAME, int SCENE, glm::mat4 E);
void genScenePixelsMonteCarlo(Image &image, int width, int height, std::shared_ptr<Camera> &camPos,
                              const std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                              const std::vector<Light> &lights, std::string FILENAME, int SCENE, glm::mat4 E);
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
