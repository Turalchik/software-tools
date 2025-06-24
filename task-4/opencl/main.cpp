#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

static const char* kernelCode = R"KERNEL(
__kernel void matMul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int N) {
    int r = get_global_id(0);
    int c = get_global_id(1);
    float acc = 0.0f;
    for (int k = 0; k < N; ++k) {
        acc += A[r * N + k] * B[k * N + c];
    }
    C[r * N + c] = acc;
}
)KERNEL";

inline void checkCL(cl_int e, const char* msg) {
    if (e != CL_SUCCESS) {
        std::cerr << msg << " (" << e << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

cl_device_id pickDevice(cl_platform_id plt) {
    cl_device_id d = nullptr;
    if (clGetDeviceIDs(plt, CL_DEVICE_TYPE_GPU, 1, &d, nullptr) == CL_SUCCESS)
        return d;
    if (clGetDeviceIDs(plt, CL_DEVICE_TYPE_CPU, 1, &d, nullptr) == CL_SUCCESS)
        return d;
    std::cerr << "No OpenCL device\n";
    std::exit(EXIT_FAILURE);
}

int main() {
    std::vector<int> dims = {10, 100, 1000, 2000};
    cl_int err;
    cl_platform_id plt;
    checkCL(clGetPlatformIDs(1, &plt, nullptr), "Platform");
    cl_device_id dev = pickDevice(plt);

    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    checkCL(err, "Context");
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, 0, &err);
    checkCL(err, "Queue");
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernelCode, nullptr, &err);
    checkCL(err, "Program");
    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t L=0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &L);
        std::vector<char> log(L); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, L, log.data(), nullptr);
        std::cerr << log.data() << "\n";
        return EXIT_FAILURE;
    }
    cl_kernel kn = clCreateKernel(prog, "matMul", &err);
    checkCL(err, "Kernel");

    for (int N : dims) {
        size_t sz = sizeof(float)*N*N;
        std::vector<float> A(N*N), B(N*N), C(N*N, 0.f);
        for (auto& x:A) x = rand()%10;
        for (auto& x:B) x = rand()%10;

        cl_mem mA = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz, A.data(), &err);
        checkCL(err, "BufA");
        cl_mem mB = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz, B.data(), &err);
        checkCL(err, "BufB");
        cl_mem mC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, nullptr, &err);
        checkCL(err, "BufC");

        checkCL(clSetKernelArg(kn,0,sizeof(mA),&mA),"Arg0");
        checkCL(clSetKernelArg(kn,1,sizeof(mB),&mB),"Arg1");
        checkCL(clSetKernelArg(kn,2,sizeof(mC),&mC),"Arg2");
        checkCL(clSetKernelArg(kn,3,sizeof(N), &N),"Arg3");

        size_t g[2]={size_t(N),size_t(N)};
        auto t0=std::chrono::high_resolution_clock::now();
        checkCL(clEnqueueNDRangeKernel(q,kn,2,nullptr,g,nullptr,0,nullptr,nullptr),"Enqueue");
        clFinish(q);
        checkCL(clEnqueueReadBuffer(q,mC,CL_TRUE,0,sz,C.data(),0,nullptr,nullptr),"Read");

        auto t1=std::chrono::high_resolution_clock::now();
        double dt=std::chrono::duration<double>(t1-t0).count();
        std::cout<<"N="<<N<<" -> "<<dt<<"s\n";

        clReleaseMemObject(mA);
        clReleaseMemObject(mB);
        clReleaseMemObject(mC);
    }

    clReleaseKernel(kn);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
