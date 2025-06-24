#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

const char* clSource = R"CLC(
__kernel void computeDerivativeX(__global const double* input,
                                 __global double* output,
                                 const int rows,
                                 const int cols,
                                 const double dx) {
    int gid = get_global_id(0);
    if (gid >= rows) return;

    for (int j = 0; j < cols; ++j) {
        int idx = gid * cols + j;

        if (j == 0)
            output[idx] = (input[gid * cols + j + 1] - input[idx]) / dx;
        else if (j == cols - 1)
            output[idx] = (input[idx] - input[gid * cols + j - 1]) / dx;
        else
            output[idx] = (input[gid * cols + j + 1] - input[gid * cols + j - 1]) / (2.0 * dx);
    }
}
)CLC";

constexpr double dx = 0.01;

double f(double x, double y) {
    return x * (sin(x) + cos(y));
}

int main() {
    std::vector<int> dimensions = {10, 100, 1000};

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;

    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) return 1;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
    if (err != CL_SUCCESS) return 1;

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return 1;

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    if (err != CL_SUCCESS) return 1;

    cl_program program = clCreateProgramWithSource(context, 1, &clSource, nullptr, &err);
    if (err != CL_SUCCESS) return 1;

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build error:\n" << log.data() << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "computeDerivativeX", &err);
    if (err != CL_SUCCESS) return 1;


    for (int size : dimensions) {
        int rows = size;
        int cols = size;
        size_t dataSize = rows * cols;

        std::vector<double> input(dataSize), output(dataSize, 0.0);

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                input[i * cols + j] = f(i * dx, j * dx);

        cl_mem inputBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(double) * dataSize, input.data(), &err);
        cl_mem outputBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                          sizeof(double) * dataSize, nullptr, &err);
        if (err != CL_SUCCESS) return 1;


        clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuf);
        clSetKernelArg(kernel, 2, sizeof(int), &rows);
        clSetKernelArg(kernel, 3, sizeof(int), &cols);
        clSetKernelArg(kernel, 4, sizeof(double), &dx);

        size_t globalSize = rows;

        auto t1 = std::chrono::high_resolution_clock::now();

        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return 1;

        clFinish(queue);

        clEnqueueReadBuffer(queue, outputBuf, CL_TRUE, 0, sizeof(double) * dataSize, output.data(), 0, nullptr, nullptr);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> delta = t2 - t1;

        std::cout << "Grid size: " << rows << "x" << cols
                  << ", Time: " << delta.count() << " seconds" << std::endl;

        clReleaseMemObject(inputBuf);
        clReleaseMemObject(outputBuf);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
