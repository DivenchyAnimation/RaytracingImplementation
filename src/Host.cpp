// Main routines for device to host communication
#include "Host.h"
#include "GPUShape.h"

using std::shared_ptr, std::vector;

void loadShapesOnDevice(std::vector<std::shared_ptr<Shape>> shapes, std::vector<Light> lights) {
	// Convert to GPUShape, switch out of using pointers, faster and easier to handle
	std::vector<GPUShape*> shapesHost;
	for (shared_ptr<Shape> shape : shapes) {
		GPUShape *gpuShape;
		gpuShape->position = shape->getPosition();
		gpuShape->rotation = shape->getRotationAxis();
		gpuShape->rotationAngle = shape->getRotationAngle();
		gpuShape->scale = shape->getScale();
		gpuShape->isReflective = shape->getIsReflective();
		// Update material properties
		gpuShape->material.setMaterialKA(shape->getMaterial()->getMaterialKA());
		gpuShape->material.setMaterialKD(shape->getMaterial()->getMaterialKD());
		gpuShape->material.setMaterialKS(shape->getMaterial()->getMaterialKS());
		gpuShape->material.setMaterialS(shape->getMaterial()->getMaterialS());
		gpuShape->material.setMaterialReflectivity(shape->getMaterial()->getMaterialReflectivity());
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