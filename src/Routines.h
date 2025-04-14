void initMaterials(std::vector<std::shared_ptr<Material>> &materials);
glm::vec3 genRayForPixel(int x, int y, int width, int height, float fov);
std::shared_ptr<Hit> computeIntersection(const Ray &ray, const std::shared_ptr<Shape> &shape, const Light &light);
void genScenePixels(Image &image, int width, int height, std::shared_ptr<Camera> &camPos,
                    const std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                    const Light &light, std::string FILENAME);

void taskOne(int width, int height, std::vector<std::shared_ptr<Material>> materials, std::vector<std::shared_ptr<Shape>> &shapes,
             std::vector<std::shared_ptr<Hit>> &hits, std::shared_ptr<Camera> &cam, std::string FILENAME);
