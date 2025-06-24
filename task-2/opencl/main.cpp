#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <stdexcept>

// Помощник для проверки ошибок OpenCL
inline void oclCheck(cl_int status, const char* stage) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string("OpenCL failed at ") + stage + ": error " + std::to_string(status));
    }
}

// Исходник ядра для редукции сумм
static const char* kKernelCode = R"KCL(
__kernel void reduce_sum(__global const int* src, __global int* dst, const int length) {
    int gid = get_global_id(0);
    int lsize = get_local_size(0);
    int lid = get_local_id(0);
    __local int accum[256];

    int partial = 0;
    for (int idx = gid; idx < length; idx += get_global_size(0)) {
        partial += src[idx];
    }
    accum[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = lsize >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            accum[lid] += accum[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        dst[get_group_id(0)] = accum[0];
    }
}
)KCL";

int main() {

    const std::vector<int> testSizes = {10, 1000, 10'000'000};
    std::mt19937 rng(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int> dist(0, 9);

    cl_int err;
    cl_platform_id platform = nullptr;
    oclCheck(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");
    cl_device_id device = nullptr;
    oclCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr), "clGetDeviceIDs");

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    oclCheck(err, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    oclCheck(err, "clCreateCommandQueueWithProperties");

    cl_program program = clCreateProgramWithSource(context, 1, &kKernelCode, nullptr, &err);
    oclCheck(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << "Build log:\n" << buildLog.data() << std::endl;
        return EXIT_FAILURE;
    }
    cl_kernel kernel = clCreateKernel(program, "reduce_sum", &err);
    oclCheck(err, "clCreateKernel");

    for (int length : testSizes) {

        std::vector<int> hostData(length);
        for (auto& x : hostData) x = dist(rng);

        cl_mem bufSrc = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(int) * length, hostData.data(), &err);
        oclCheck(err, "clCreateBuffer(src)");
        const size_t localSize = 256;
        size_t groupCount = (length + localSize - 1) / localSize;
        cl_mem bufDst = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       sizeof(int) * groupCount, nullptr, &err);
        oclCheck(err, "clCreateBuffer(dst)");

        oclCheck(clSetKernelArg(kernel, 0, sizeof(bufSrc), &bufSrc), "clSetKernelArg(0)");
        oclCheck(clSetKernelArg(kernel, 1, sizeof(bufDst), &bufDst), "clSetKernelArg(1)");
        oclCheck(clSetKernelArg(kernel, 2, sizeof(length), &length), "clSetKernelArg(2)");

        size_t globalSize = localSize * groupCount;

        auto t0 = std::chrono::high_resolution_clock::now();
        oclCheck(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                        0, nullptr, nullptr), "clEnqueueNDRangeKernel");
        std::vector<int> partials(groupCount);
        oclCheck(clEnqueueReadBuffer(queue, bufDst, CL_TRUE, 0,
                                     sizeof(int) * groupCount, partials.data(), 0, nullptr, nullptr),
                 "clEnqueueReadBuffer");
        auto t1 = std::chrono::high_resolution_clock::now();

        long total = 0;
        for (int v : partials) total += v;
        std::chrono::duration<double> elapsed = t1 - t0;

        std::cout << "Size=" << length
                  << " Sum=" << total
                  << " Time=" << elapsed.count() << "s\n";

        clReleaseMemObject(bufSrc);
        clReleaseMemObject(bufDst);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}