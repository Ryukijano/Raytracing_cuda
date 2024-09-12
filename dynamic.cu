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
#define MAX_DEPTH 5
#define SAMPLES_PER_PIXEL 10
#define FRAMES 180

struct Sphere {
    float3 center;
    float radius;
    float3 color;
    float reflectivity;
    float3 velocity;
};

struct Ray {
    float3 origin;
    float3 direction;
};

// Device functions
__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3& a, float t) {
    return make_float3(a.x * t, a.y * t, a.z * t);
}

__device__ float3 operator*(float t, const float3& a) {
    return a * t;
}

__device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(const float3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return v * (1.0f / len);
}

__device__ float3 reflect(const float3& v, const float3& n) {
    return v - 2 * dot(v, n) * n;
}

__device__ bool hit_sphere(const Sphere& sphere, const Ray& ray, float t_min, float t_max, float& t) {
    float3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;
    
    if (discriminant > 0) {
        float temp = (-b - sqrtf(discriminant)) / (2.0f * a);
        if (temp < t_max && temp > t_min) {
            t = temp;
            return true;
        }
        temp = (-b + sqrtf(discriminant)) / (2.0f * a);
        if (temp < t_max && temp > t_min) {
            t = temp;
            return true;
        }
    }
    return false;
}

__device__ float3 color(const Ray& r, Sphere* spheres, curandState* local_rand_state, int depth) {
    float t;
    float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);
    Ray current_ray = r;

    for (int d = 0; d < depth; d++) {
        int hit_idx = -1;
        float t_closest = FLT_MAX;

        for (int i = 0; i < SPHERES; i++) {
            if (hit_sphere(spheres[i], current_ray, 0.001f, t_closest, t)) {
                hit_idx = i;
                t_closest = t;
            }
        }

        if (hit_idx != -1) {
            float3 hit_point = current_ray.origin + t_closest * current_ray.direction;
            float3 normal = normalize(hit_point - spheres[hit_idx].center);
            
            // Diffuse reflection
            float3 target = hit_point + normal + make_float3(
                curand_uniform(local_rand_state) - 0.5f,
                curand_uniform(local_rand_state) - 0.5f,
                curand_uniform(local_rand_state) - 0.5f
            );
            
            float3 diffuse_dir = normalize(target - hit_point);
            
            // Specular reflection
            float3 reflected_dir = reflect(current_ray.direction, normal);
            
            // Mix diffuse and specular based on reflectivity
            float3 scatter_dir = normalize(
                (1.0f - spheres[hit_idx].reflectivity) * diffuse_dir +
                spheres[hit_idx].reflectivity * reflected_dir
            );
            
            current_ray = Ray{hit_point, scatter_dir};
            attenuation = attenuation * spheres[hit_idx].color;
        } else {
            // Sky color
            float t = 0.5f * (normalize(current_ray.direction).y + 1.0f);
            float3 sky_color = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
            return attenuation * sky_color;
        }
    }

    return make_float3(0.0f, 0.0f, 0.0f); // Exceeded recursion depth
}

__global__ void render(Sphere* spheres, unsigned char* image, int frame, curandState* rand_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * WIDTH + x) * 3;

    if (x >= WIDTH || y >= HEIGHT) return;

    curandState local_rand_state = rand_state[y * WIDTH + x];

    float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
        float u = float(x + curand_uniform(&local_rand_state)) / float(WIDTH);
        float v = float(y + curand_uniform(&local_rand_state)) / float(HEIGHT);

        Ray r;
        r.origin = make_float3(0.0f, 0.0f, -10.0f);
        r.direction = normalize(make_float3(u - 0.5f, v - 0.5f, 1.0f));

        pixel_color = pixel_color + color(r, spheres, &local_rand_state, MAX_DEPTH);
    }

    pixel_color = pixel_color * (1.0f / SAMPLES_PER_PIXEL);

    // Apply simple tone mapping (gamma correction)
    pixel_color.x = sqrtf(pixel_color.x);
    pixel_color.y = sqrtf(pixel_color.y);
    pixel_color.z = sqrtf(pixel_color.z);

    image[idx] = static_cast<unsigned char>(fminf(pixel_color.x, 1.0f) * 255);
    image[idx + 1] = static_cast<unsigned char>(fminf(pixel_color.y, 1.0f) * 255);
    image[idx + 2] = static_cast<unsigned char>(fminf(pixel_color.z, 1.0f) * 255);
}

__global__ void update_spheres(Sphere* spheres, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= SPHERES) return;

    spheres[idx].center = spheres[idx].center + spheres[idx].velocity * dt;

    // Simple boundary check and bounce
    if (fabsf(spheres[idx].center.x) > 5.0f) spheres[idx].velocity.x *= -1;
    if (fabsf(spheres[idx].center.y) > 5.0f) spheres[idx].velocity.y *= -1;
    if (fabsf(spheres[idx].center.z) > 5.0f) spheres[idx].velocity.z *= -1;
}

__global__ void init_rand_state(curandState* rand_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = y * WIDTH + x;
    curand_init(1984, idx, 0, &rand_state[idx]);
}

void initializeSpheres(Sphere* spheres) {
    for (int i = 0; i < SPHERES; ++i) {
        spheres[i].center = make_float3(
            static_cast<float>(rand()) / RAND_MAX * 10 - 5,
            static_cast<float>(rand()) / RAND_MAX * 10 - 5,
            static_cast<float>(rand()) / RAND_MAX * 10 - 5
        );
        spheres[i].radius = static_cast<float>(rand()) / RAND_MAX * 0.5 + 0.1;
        spheres[i].color = make_float3(
            static_cast<float>(rand()) / RAND_MAX,
            static_cast<float>(rand()) / RAND_MAX,
            static_cast<float>(rand()) / RAND_MAX
        );
        spheres[i].reflectivity = static_cast<float>(rand()) / RAND_MAX * 0.5;
        spheres[i].velocity = make_float3(
            (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2,
            (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2,
            (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2
        );
    }
}

void saveImage(const unsigned char* image, int frame) {
    char filename[20];
    snprintf(filename, sizeof(filename), "output_%03d.ppm", frame);
    std::ofstream file(filename);
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
    curandState* d_rand_state;
    cudaMalloc(&d_spheres, SPHERES * sizeof(Sphere));
    cudaMalloc(&d_image, WIDTH * HEIGHT * 3);
    cudaMalloc(&d_rand_state, WIDTH * HEIGHT * sizeof(curandState));

    // Copy spheres from host to device
    cudaMemcpy(d_spheres, h_spheres, SPHERES * sizeof(Sphere), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Initialize random states
    init_rand_state<<<blocksPerGrid, threadsPerBlock>>>(d_rand_state);

    // Render frames
    for (int frame = 0; frame < FRAMES; frame++) {
        // Update sphere positions
        update_spheres<<<(SPHERES + 255) / 256, 256>>>(d_spheres, 0.1f);

        // Render the frame
        render<<<blocksPerGrid, threadsPerBlock>>>(d_spheres, d_image, frame, d_rand_state);

        // Copy result from device to host
        cudaMemcpy(h_image, d_image, WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);

        // Save the image
        saveImage(h_image, frame);

        std::cout << "Rendered frame " << frame + 1 << " of " << FRAMES << std::endl;
    }

    // Free device memory
    cudaFree(d_spheres);
    cudaFree(d_image);
    cudaFree(d_rand_state);

    // Free host memory
    free(h_spheres);
    free(h_image);

    std::cout << "Ray tracing animation completed successfully!" << std::endl;

    return 0;
}