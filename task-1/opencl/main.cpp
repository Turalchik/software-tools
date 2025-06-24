#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <iostream>
#include <vector>

static const char* kKernelSrc = R"CLC(
__kernel void hello_opencl(__global int* out, int count) {
    int gid = get_global_id(0);
    out[gid] = gid;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
)CLC";

const char* clErrorString(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        default: return "UNKNOWN_ERROR";
    }
}

int main() {
    cl_int err;
    cl_uint numPlatforms = 0;

    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "ERROR: No OpenCL platforms found (" << clErrorString(err) << ")\n";
        return 1;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    cl_device_type devType = CL_DEVICE_TYPE_GPU;
    cl_device_id device = nullptr;

    for (auto& plat : platforms) {
        err = clGetDeviceIDs(plat, devType, 1, &device, nullptr);
        if (err == CL_SUCCESS) {
            std::cout << "Using GPU device on platform.\n";
            break;
        }
    }
    if (!device) {
        std::cout << "GPU not found, falling back to CPU.\n";
        devType = CL_DEVICE_TYPE_CPU;
        for (auto& plat : platforms) {
            err = clGetDeviceIDs(plat, devType, 1, &device, nullptr);
            if (err == CL_SUCCESS) {
                std::cout << "Using CPU device on platform.\n";
                break;
            }
        }
    }
    if (!device) {
        std::cerr << "ERROR: No suitable OpenCL device found.\n";
        return 1;
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateContext failed: " << clErrorString(err) << "\n";
        return 1;
    }
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateCommandQueue failed: " << clErrorString(err) << "\n";
        clReleaseContext(context);
        return 1;
    }

    const int kNumItems = 4;
    size_t bufSize = sizeof(int) * kNumItems;
    cl_mem buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufSize, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateBuffer failed: " << clErrorString(err) << "\n";
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kKernelSrc, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateProgramWithSource failed: " << clErrorString(err) << "\n";
        return 1;
    }
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logLen = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen);
        std::vector<char> log(logLen);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logLen, log.data(), nullptr);
        std::cerr << "Build error:\n" << log.data() << "\n";
        clReleaseProgram(program);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "hello_opencl", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateKernel failed: " << clErrorString(err) << "\n";
        return 1;
    }
    clSetKernelArg(kernel, 0, sizeof(buf), &buf);
    clSetKernelArg(kernel, 1, sizeof(kNumItems), &kNumItems);

    size_t global = kNumItems;
    size_t local  = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueNDRangeKernel failed: " << clErrorString(err) << "\n";
    }

    std::vector<int> out(kNumItems);
    clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bufSize, out.data(), 0, nullptr, nullptr);

    for (int i = 0; i < kNumItems; ++i) {
        std::cout << "Hello from work-item " << out[i] << "\n";
    }

    clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
