//
// Created by dylan on 1/22/23.
//

#ifndef SIMPLEPATHTRACER_UTILFUNCTIONS_CUH
#define SIMPLEPATHTRACER_UTILFUNCTIONS_CUH

#define CUDA_BS_1D 256
#define REDUCE_BLOCK_SIZE 1024
#define WARP_SIZE 32
#define REDUCE_BUFFER_SIZE 8192

#include <cuda_runtime.h>
#include <iostream>
#include "../Model/TrigModel.cuh"

using namespace std;
namespace dylanrt {

    typedef unsigned int uint32;
    extern void* PUBLIC_REDUCTION_BUFFER;

    __host__ void assertCudaError();

    __host__ void reorientModel(TrigModel* model);

    void solveTirgNormals(triangle* triangles, float3* vertices, unsigned int numTrigs);

    float maxVal(float* imagePlane, int resolutionX, int resolutionY);

    float minVal(float* imagePlane, int resolutionX, int resolutionY);

    //normalize all pixels between 1 and 256
    void createImage(float* imagePlane, unsigned char* outPixels, int resolutionX, int resolutionY);


} // dylanrt

#endif //SIMPLEPATHTRACER_UTILFUNCTIONS_CUH
