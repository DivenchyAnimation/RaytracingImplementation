// Main routines for device to host communication
#include "Host.h"

using std::shared_ptr, std::vector, std::make_shared;

// Define E mat

// E =
//
//     1.5000         0         0    0.3000
//          0    1.4095   -0.5130   -1.5000
//          0    0.5130    1.4095         0
//          0         0         0    1.0000
mat4 E(
	vec4(1.5f, 0.0f, 0.0f, 0.3f), 
	vec4(0.0f, 1.4095f, -0.5130, -1.500), 
	vec4(0.0f, 0.5130f, 1.4095f, 0.0f), 
	vec4(0.0f, 0.0f, 0.0, 1.0f));

void loadShapesOnDevice(std::vector<std::shared_ptr<Shape>> shapes, GPUShape **device_shapesPtrs) {
	// Convert to GPUShape, switch out of using pointers, faster and easier to handle
	std::vector<GPUShape*> shapesHost;
	for (shared_ptr<Shape> shape : shapes) {
		GPUShape *gpuShape = (GPUShape*)malloc(sizeof(gpuShape));
		gpuShape->position = vec3(shape->getPosition().x, shape->getPosition().y, shape->getPosition().z);
		gpuShape->rotation = vec3(shape->getRotationAxis().x, shape->getRotationAxis().y, shape->getRotationAxis().z);
		gpuShape->rotationAngle = shape->getRotationAngle();
		gpuShape->scale = vec3(shape->getScale().x, shape->getScale().y, shape->getScale().z);
		gpuShape->isReflective = shape->getIsReflective();
		// Update material properties
		GPUMaterial *shapeMaterialPtr = (GPUMaterial*)malloc(sizeof(GPUMaterial));
		GPUMaterial shapeMaterial;
		shapeMaterial.setMaterialKA(vec3(shape->getMaterial()->getMaterialKA().x, shape->getMaterial()->getMaterialKA().y, shape->getMaterial()->getMaterialKA().z));
		shapeMaterial.setMaterialKD(vec3(shape->getMaterial()->getMaterialKD().x, shape->getMaterial()->getMaterialKD().y, shape->getMaterial()->getMaterialKD().z));
		shapeMaterial.setMaterialKS(vec3(shape->getMaterial()->getMaterialKS().x, shape->getMaterial()->getMaterialKS().y, shape->getMaterial()->getMaterialKS().z));
		shapeMaterial.setMaterialS(shape->getMaterial()->getMaterialS());
		shapeMaterial.setMaterialReflectivity(shape->getMaterial()->getMaterialReflectivity());
		*shapeMaterialPtr = shapeMaterial;
		gpuShape->material = shapeMaterialPtr;
		if (shape->getType() == ShapeType::SPHERE) {
			gpuShape->type = GPUShapeType::SPHERE;
			shared_ptr<Sphere> sphere = std::static_pointer_cast<Sphere>(shape);
			gpuShape->data.SPHERE.radius = sphere->getRadius();
		}

		// After transferring to GPUShape add to the new GPUShape vec
		shapesHost.push_back(gpuShape);
	}

	// Allocate GPUShapes to device, C-Style
	cudaMalloc(&device_shapesPtrs, shapesHost.size() * sizeof(GPUShape*));
	cudaMemcpy(device_shapesPtrs, shapesHost.data(), shapesHost.size() * sizeof(GPUShape*), cudaMemcpyHostToDevice);
}

void loadLightsOnDevice(std::vector<Light> lights, GPULight *device_lights) {
	std::vector<GPULight> lightsHost;
	for (Light light : lights) {
		GPULight gpuLight;
		gpuLight.pos = vec3(light.pos.x, light.pos.y, light.pos.z);
		gpuLight.ringRadius = light.ringRadius;
		gpuLight.baseAngle = light.baseAngle;
		gpuLight.color = vec3(light.color.x, light.color.y, light.color.z);
		gpuLight.intensity = light.intensity;
		lightsHost.push_back(gpuLight);
	}

	// Allocate GPULights to device
	cudaMalloc(&device_lights, lightsHost.size() * sizeof(GPULight));
	cudaMemcpy(device_lights, lightsHost.data(), lightsHost.size() * sizeof(GPULight), cudaMemcpyHostToDevice);
}

// HA -> Hardware Accelerated
void HAsceneOne(int width, int height, std::vector<std::shared_ptr<Material>> materials,
              std::vector<std::shared_ptr<Shape>> &shapes, GPUShape **device_shapesPtrs, GPULight *device_lights, std::shared_ptr<Camera> &cam,
              char *FILENAME) {
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

	loadShapesOnDevice(shapes, device_shapesPtrs);
	loadLightsOnDevice(lights, device_lights);
	// Create device friendly camera, and to device friendly structs
	GPUCamera *gpuCam;
	gpuCam->position = vec3(cam->getPosition().x, cam->getPosition().y, cam->getPosition().z);
	gpuCam->fovy = cam->getFOVY();
	gpuCam->setTarget(vec3(cam->getTarget().x, cam->getTarget().y, cam->getTarget().z));
	
	 
	KernelGenScenePixels(image, width, height, gpuCam, device_shapesPtrs, 3, device_lights, 1, FILENAME, 1, E);
}