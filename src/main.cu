#include <iostream>
#include <cuda_runtime.h>


__global__ void addOne(int *x) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    x[idx] += 1;
}

int main() {
    // 1. Query device count
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
            << cudaGetErrorString(err) << "\n";
        return 1;
    }
    std::cout << "Found " << deviceCount << " CUDA device(s).\n";

    if (deviceCount == 0) return 0;

    // 2. Simple vector add test
    const int N = 16;
    int  h_data[N];
    for (int i = 0; i < N; ++i) h_data[i] = i;

    int *d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int),
        cudaMemcpyHostToDevice);

    addOne << <1, N >> > (d_data);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, N * sizeof(int),
        cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // 3. Verify
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] != i + 1) {
            ok = false; break;
        }
    }
    std::cout << (ok ? "PASS" : "FAIL") << "\n";

    return ok ? 0 : 2;
}