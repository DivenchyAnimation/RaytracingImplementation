// Main routines for device to host communication
#include "Host.h"

using std::shared_ptr, std::vector;

void loadShapesOnDevice(std::vector<std::shared_ptr<Shape>> shapes, std::vector<Light> lights) {
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
	GPUShape **device_shapesPtrs = NULL;
	cudaMalloc(&device_shapesPtrs, shapesHost.size() * sizeof(GPUShape*));
	cudaMemcpy(device_shapesPtrs, shapesHost.data(), shapesHost.size() * sizeof(GPUShape*), cudaMemcpyHostToDevice);
}