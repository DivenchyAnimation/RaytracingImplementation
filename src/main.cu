#include "pch.cuh"
#include "Host.h"
#include "HostLauncher.h"
#include <iostream>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

char *FILENAME;
int SCENE;
int width, height;
int nShapes;
int nLights;
std::vector<GPUShape> hostShapes;
std::vector<GPULight> hostLights;
GPUShape *device_shapes;
GPUShape **device_shapesPtrs;
GPULight *device_lights;
GPUCamera *device_cam;
GPUMaterial *materials;
GPUMaterial *device_materials;

extern __global__ void fillRedKernel(unsigned char *, int);

int main(int arc, char** argv) {

	if (arc != 4) {
		printf("Usage: ./A6 <SCENE> <IMAGE SIZE> <OUTFILENAME>");
		return 1;
	}

	// Init user input
	SCENE = atoi(argv[1]);
	width = atoi(argv[2]);
	height = width;
	FILENAME = argv[3];
	mat4 IdMat = mat4();

	// Load geometry
	std::vector<float> posBuf; // list of vertex positions
	std::vector<float> zBuf;   // list of z-vals for image
	std::vector<float> norBuf; // list of vertex normals
	std::vector<float> texBuf; // list of vertex texture coords
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapesTINY;
	std::vector<tinyobj::material_t> materialsTINY;
	std::string warnStr, errStr;
	std::string meshName = "../resources/bunny.obj";

	//Load the OBJ file
	if (SCENE == 6 || SCENE == 7) {
		bool rc = tinyobj::LoadObj(&attrib, &shapesTINY, &materialsTINY, &warnStr, &errStr, meshName.c_str());
		if (!rc) {
			std::cerr << errStr << std::endl;
		}
		else {
			// Some OBJ files have different indices for vertex positions, normals,
			// and texture coordinates. For example, a cube corner vertex may have
			// three different normals. Here, we are going to duplicate all such
			// vertices.
			// Loop over shapes
			for (size_t s = 0; s < shapesTINY.size(); s++) {
				// Loop over faces (polygons)
				size_t index_offset = 0;
				for (size_t f = 0; f < shapesTINY[s].mesh.num_face_vertices.size(); f++) {
					size_t fv = shapesTINY[s].mesh.num_face_vertices[f];
					// Loop over vertices in the face.
					for (size_t v = 0; v < fv; v++) {
						// access to vertex
						tinyobj::index_t idx = shapesTINY[s].mesh.indices[index_offset + v];
						posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
						posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
						posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 2]);
						if (!attrib.normals.empty()) {
							norBuf.push_back(attrib.normals[3 * idx.normal_index + 0]);
							norBuf.push_back(attrib.normals[3 * idx.normal_index + 1]);
							norBuf.push_back(attrib.normals[3 * idx.normal_index + 2]);
						}
						if (!attrib.texcoords.empty()) {
							texBuf.push_back(attrib.texcoords[2 * idx.texcoord_index + 0]);
							texBuf.push_back(attrib.texcoords[2 * idx.texcoord_index + 1]);
						}
					}
					index_offset += fv;
					// per-face material (IGNORE)
					shapesTINY[s].mesh.material_ids[f];
				}
			}
		}
		std::cout << "Number of vertices: " << posBuf.size() / 3 << std::endl;
	}

	// Create Image
	Image image(width, height);

	// Allocate memory on the device
	unsigned char *d_pixels = NULL;
	size_t numPixels = width * height;
	size_t bufferSize = numPixels * sizeof(unsigned char) * 3;
	cudaMalloc(&d_pixels, bufferSize);
	// Launch kernel for scene , thanks GPT for these two vals
	int numThreads = 256;
	int blocks = (numPixels + numThreads - 1) / numThreads;

	switch (SCENE) {
	case 1:
	case 2: {
		HAsceneOne(blocks, numThreads, d_pixels, numPixels, width, height, device_materials, materials, device_shapes, device_shapesPtrs, nShapes, device_lights, nLights, device_cam, IdMat);
		break;
	};
	case 3: {
		HAsceneThree(blocks, numThreads, d_pixels, numPixels, width, height, device_materials, materials, device_shapes, device_shapesPtrs, nShapes, device_lights, nLights, device_cam, IdMat);
		break;
	};
	case 4:
	case 5: {
		HAsceneReflections(blocks, numThreads, d_pixels, numPixels, width, height, device_materials, materials, device_shapes, device_shapesPtrs, nShapes, device_lights, nLights, device_cam, IdMat);
		break;
	};
	case 6: {
		HAsceneMesh(posBuf, norBuf, texBuf, blocks, numThreads, d_pixels, numPixels, width, height, device_materials, materials, device_shapes, device_shapesPtrs, nShapes, device_lights, nLights, device_cam, IdMat);
		break;
	}
	case 7: {
		HAsceneMeshTransform(posBuf, norBuf, texBuf, blocks, numThreads, d_pixels, numPixels, width, height, device_materials, materials, device_shapes, device_shapesPtrs, nShapes, device_lights, nLights, device_cam, IdMat);
		break;
	}
	case 8: {
		HAsceneCameraTransform(blocks, numThreads, d_pixels, numPixels, width, height, device_materials, materials, device_shapes, device_shapesPtrs, nShapes, device_lights, nLights, device_cam, IdMat);
		break;
	}
	}
	KernelGenScenePixels << <blocks, numThreads >> > (d_pixels, numPixels, width, height, device_cam, device_shapesPtrs, nShapes, device_lights, nLights, SCENE, IdMat);
	cudaDeviceSynchronize();

	// 3) Copy back into the Image's vector
	cudaMemcpy(image.getPixels(), d_pixels, bufferSize, cudaMemcpyDeviceToHost);
	cudaFree(d_pixels);

	// 4) Write out
	image.writeToFile(FILENAME);
	return 0;
}