//
// Created by dylan on 2/7/23.
//

#include "Transform.cuh"



namespace dylanrt{

    __global__ void transformD(float3* vertices, float3* screenVertices, float3* screenSolved, bool* inrange, CameraFrame* cam){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

        //store the camera frame in registers
        float3 topLeft = cam->imgTopLeft;
        float3 topRight = cam->imgTopRight;
        float3 bottomLeft = cam->imgBottomLeft;
        float3 e = cam->positionE;
        //transform the vertex to screen space
        float3 d = subtract3d(vertices[idx], e);

        //inline solver uses 8 registers
        float3 solved = cramerIntersectSolver(e,d, topLeft, topRight, bottomLeft);

        //if the intersection is within the image plane, store the intersection
        screenSolved[idx] = solved;

        float3 out = add3d(topLeft, scale3d(subtract3d(topRight,topLeft), solved.x));
        out = add3d(out, scale3d(subtract3d(bottomLeft,topLeft), solved.y));

        //store the screen space vertex
        screenVertices[idx] = out;
        inrange[idx] = rectInRange(solved.x, solved.y);
    }

    void transform(float3* vertices, float3* screenVertices, float3* screenSolved, CameraFrame* cam, bool* inrange, int numVertices){
        //calculate the number of blocks and threads
        int numThreads = 512;
        int numBlocks = (numVertices + numThreads - 1) / numThreads;

        //call the kernel
        transformD<<<numBlocks, numThreads>>>(vertices, screenVertices, screenSolved, inrange,cam);
        cudaDeviceSynchronize();
        assertCudaError();
    }
}