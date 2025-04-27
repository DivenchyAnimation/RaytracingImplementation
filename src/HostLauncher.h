#pragma once
#include <vector>
#include <memory>
#include "GPUShape.h"
#include "GPULight.h"
#include "GPUMaterial.h"
#include "GPUCamera.h"
#include "Kernels.h"
#include "GPUVecOps.cuh"
#include <cuda_runtime.h>
#include "Host.h"

void loadHostShapes(std::vector<GPUShape> &hostShapes, GPUMaterial *materials);

void HAsceneOne(int blocks, int numThreads, unsigned char *&d_pixels, int numPixels, int width, int height, GPUMaterial *&device_materials, GPUMaterial *&materials,
    GPUShape *&device_shapes, GPUShape **&device_shapesPtrs, int &nShapes, GPULight *&device_lights, int &nLights, GPUCamera *&device_cam, mat4 &E);

void HAsceneThree(int blocks, int numThreads, unsigned char *&d_pixels, int numPixels, int width, int height, GPUMaterial *&device_materials, GPUMaterial *&materials,
    GPUShape *&device_shapes, GPUShape **&device_shapesPtrs, int &nShapes, GPULight *&device_lights, int &nLights, GPUCamera *&device_cam, mat4 &E);

void HAsceneReflections(int blocks, int numThreads, unsigned char *&d_pixels, int numPixels, int width, int height, GPUMaterial *&device_materials, GPUMaterial *&materials,
    GPUShape *&device_shapes, GPUShape **&device_shapesPtrs, int &nShapes, GPULight *&device_lights, int &nLights, GPUCamera *&device_cam, mat4 &E);

void HAsceneMesh(std::vector<float> &posBuf, std::vector<float> &norBuf, std::vector<float> &texBuf, int blocks, int numThreads, unsigned char *&d_pixels, int numPixels, int width, int height, GPUMaterial *&device_materials, GPUMaterial *&materials,
    GPUShape *&device_shapes, GPUShape **&device_shapesPtrs, int &nShapes, GPULight *&device_lights, int &nLights, GPUCamera *&device_cam, mat4 &E);

void HAsceneMeshTransform(std::vector<float> &posBuf, std::vector<float> &norBuf, std::vector<float> &texBuf, int blocks, int numThreads, unsigned char *&d_pixels, int numPixels, int width, int height, GPUMaterial *&device_materials, GPUMaterial *&materials,
    GPUShape *&device_shapes, GPUShape **&device_shapesPtrs, int &nShapes, GPULight *&device_lights, int &nLights, GPUCamera *&device_cam, mat4 &E);

void HAsceneCameraTransform(int blocks, int numThreads, unsigned char *&d_pixels, int numPixels, int width, int height, GPUMaterial *&device_materials, GPUMaterial *&materials,
    GPUShape *&device_shapes, GPUShape **&device_shapesPtrs, int &nShapes, GPULight *&device_lights, int &nLights, GPUCamera *&device_cam, mat4 &E);
