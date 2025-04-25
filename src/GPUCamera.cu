#include "GPUCamera.h"
// -Z forward, Y up
GPUCamera::GPUCamera()
    : aspect(1.0f), fovy(GPUradians(45.0f)), znear(0.1f), zfar(1000.0f), rotations(0.0, 0.0), translations(0.0f, 0.0f, -5.0f),
      rfactor(0.01f), tfactor(0.001f), sfactor(0.005f), position(0.0f, 0.0f, 5.0f) {}

GPUCamera::~GPUCamera() {}
