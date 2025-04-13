glm::vec3 genRayForPixel(int x, int y, int width, int height, float fov);
glm::vec3 computeIntersection(const Ray &ray, const std::shared_ptr<Shape> &shape, const Light &light);
void genPixelsSceneOne(Image &image, int width, int height, std::shared_ptr<Camera> &camPos, const std::shared_ptr<Shape> &shape,
                       const Light &light, std::string FILENAME);

void taskOne(int width, int height, std::shared_ptr<Camera> &cam, std::string FILENAME);
