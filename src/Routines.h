void initMaterials(std::vector<std::shared_ptr<Material>> &materials);
glm::vec3 genRayForPixel(int x, int y, int width, int height, float fov);
std::shared_ptr<Hit> computeIntersectionSphere(const Ray &ray, const std::shared_ptr<Shape> &shape, const glm::mat4 modelMat,
                                               const glm::mat4 modelMatInv, const std::vector<Light> &lights);
std::shared_ptr<Hit> computeIntersectionPlane(const Ray &ray, const std::shared_ptr<Shape> &shape, const glm::mat4 modelMat,
                                              const glm::mat4 modelMatInv, const std::vector<Light> &lights);
bool isInShadow(std::shared_ptr<Hit> nearestHit, const std::vector<Light> &lights,
                const std::vector<std::shared_ptr<Shape>> &shapes);
void genScenePixels(Image &image, int width, int height, std::shared_ptr<Camera> &camPos,
                    const std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                    const std::vector<Light> &lights, std::string FILENAME);

void sceneOne(int width, int height, std::vector<std::shared_ptr<Material>> materials,
              std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits, std::shared_ptr<Camera> &cam,
              std::string FILENAME);
void sceneThree(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                std::shared_ptr<Camera> &cam, std::string FILENAME);
