#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cmath>

#define WIDTH 800
#define HEIGHT 600
#define SPHERES 20

struct Sphere {
    float x, y, z, radius;
    float r, g, b;
};

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(const float3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ bool hit_sphere(const Sphere& sphere, const float3& ray_origin, const float3& ray_dir, float& t) {
    float3 oc = make_float3(ray_origin.x - sphere.x, ray_origin.y - sphere.y, ray_origin.z - sphere.z);
    float a = dot(ray_dir, ray_dir);
    float b = 2.0f * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant > 0) {
        t = (-b - sqrtf(discriminant)) / (2.0f * a);
        return true;
    }
    return false;
}

__global__ void render(Sphere* spheres, unsigned char* image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * WIDTH + x) * 3;

    if (x >= WIDTH || y >= HEIGHT) return;

    float3 ray_origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 ray_dir = make_float3((x - WIDTH / 2.0f) / WIDTH, (y - HEIGHT / 2.0f) / HEIGHT, 1.0f);
    ray_dir = normalize(ray_dir);

    float t_min = 1e20;
    int hit_idx = -1;
    bool hit = false;

    for (int i = 0; i < SPHERES; ++i) {
        float t;
        if (hit_sphere(spheres[i], ray_origin, ray_dir, t) && t < t_min) {
            t_min = t;
            hit_idx = i;
            hit = true;
        }
    }

    if (hit) {
        image[idx] = static_cast<unsigned char>(spheres[hit_idx].r * 255);
        image[idx + 1] = static_cast<unsigned char>(spheres[hit_idx].g * 255);
        image[idx + 2] = static_cast<unsigned char>(spheres[hit_idx].b * 255);
    } else {
        image[idx] = 0;
        image[idx + 1] = 0;
        image[idx + 2] = 0;
    }
}

void initializeSpheres(Sphere* spheres) {
    for (int i = 0; i < SPHERES; ++i) {
        spheres[i].x = static_cast<float>(rand()) / RAND_MAX * 10 - 5;
        spheres[i].y = static_cast<float>(rand()) / RAND_MAX * 10 - 5;
        spheres[i].z = static_cast<float>(rand()) / RAND_MAX * 10 - 5;
        spheres[i].radius = static_cast<float>(rand()) / RAND_MAX * 0.5 + 0.1;
        spheres[i].r = static_cast<float>(rand()) / RAND_MAX;
        spheres[i].g = static_cast<float>(rand()) / RAND_MAX;
        spheres[i].b = static_cast<float>(rand()) / RAND_MAX;
    }
}

void saveImage(const unsigned char* image) {
    std::ofstream file("output.ppm");
    file << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    file.write(reinterpret_cast<const char*>(image), WIDTH * HEIGHT * 3);
    file.close();
}

int main() {
    srand(time(0));

    // Allocate host memory
    Sphere* h_spheres = (Sphere*)malloc(SPHERES * sizeof(Sphere));
    unsigned char* h_image = (unsigned char*)malloc(WIDTH * HEIGHT * 3);

    // Initialize spheres
    initializeSpheres(h_spheres);

    // Allocate device memory
    Sphere* d_spheres;
    unsigned char* d_image;
    cudaMalloc(&d_spheres, SPHERES * sizeof(Sphere));
    cudaMalloc(&d_image, WIDTH * HEIGHT * 3);

    // Copy spheres from host to device
    cudaMemcpy(d_spheres, h_spheres, SPHERES * sizeof(Sphere), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the render kernel
    render<<<blocksPerGrid, threadsPerBlock>>>(d_spheres, d_image);

    // Copy result from device to host
    cudaMemcpy(h_image, d_image, WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);

    // Save the image
    saveImage(h_image);

    // Free device memory
    cudaFree(d_spheres);
    cudaFree(d_image);

    // Free host memory
    free(h_spheres);
    free(h_image);

    std::cout << "Ray tracing completed successfully!" << std::endl;

    return 0;
}
