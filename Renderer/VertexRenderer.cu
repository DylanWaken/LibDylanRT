//
// Created by dylan on 1/21/23.
//

#include <iostream>
#include "VertexRenderer.cuh"

namespace dylanrt {


    /*
     * [a_x - b_x  a_x - c_x  d_x] [beta ]   [a_x - e_x]
     * [a_y - b_y  a_y - c_y  d_y] [gamma] = [a_y - e_y]
     * [a_z - b_z  a_z - c_z  d_z] [t    ]   [a_z - e_z]
     */
    __device__ __forceinline__ float3 cramerIntersectSolver(float3 e, float3 d,
                                                            float3 a, float3 b,
                                                            float3 c){
        //determinant of left hand side matrix
        float M = (a.x - b.x) * ((a.y - c.y) * (d.z) - (d.y) * (a.z - c.z))
                + (a.y - b.y) * ((d.x) * (a.z - c.z) - (a.x - c.x) * (d.z))
                + (a.z - b.z) * ((a.x - c.x) * (d.y) - (d.x) * (a.y - c.y));

        //first cramer determinant
        float beta = (a.x - e.x) * ((a.y - c.y) * (d.z) - (a.z - c.z) * (d.y))
                   + (a.y - e.y) * ((a.z - c.z) * (d.x) - (a.x - c.x) * (d.z))
                   + (a.z - e.z) * ((a.x - c.x) * (d.y) - (a.y - c.y) * (d.x));
        beta /= M;

        //second cramer determinant
        float gamma = (a.x - b.x) * ((a.y - e.y) * (d.z) - (a.z - e.z) * (d.y))
                    + (a.y - b.y) * ((a.z - e.z) * (d.x) - (a.x - e.x) * (d.z))
                    + (a.z - b.z) * ((a.x - e.x) * (d.y) - (a.y - e.y) * (d.x));
        gamma /= M;

        //third cramer determinant
        float t = (a.x - b.x) * ((a.y - c.y) * (a.z - e.z) - (a.z - c.z) * (a.y - e.y))
                + (a.y - b.y) * ((a.z - c.z) * (a.x - e.x) - (a.x - c.x) * (a.z - e.z))
                + (a.z - b.z) * ((a.x - c.x) * (a.y - e.y) - (a.y - c.y) * (a.x - e.x));
        t /= M;

        return make_float3(beta, gamma, t);
    }

    __device__ __forceinline__ bool rectInRange(float beta,float gamma){
        return beta >= 0 && beta <= 1 && gamma >= 0 && gamma <= 1;
    }

    template<const int BM, const int SHR_BATCH_DEPTH>
    __global__ void renderVerticesD(float3 *vertices, uint32 numVertices,
                                    CameraFrame* cameraFrame, float* imagePlane){

        uint32 begin = numVertices / gridDim.x * blockIdx.x;
        uint32 end = blockIdx.x == gridDim.x - 1 ? numVertices : numVertices / gridDim.x * (blockIdx.x + 1);

        //store cameraframe in registers
        CameraFrame cameraFrameR[1] = {*cameraFrame};

        //store the pixel coords that are going to be illuminated
        __shared__ int2 pixelCoords[BM * SHR_BATCH_DEPTH];

        //store the vertices to process
        __shared__ float3 verticesS[2][BM * SHR_BATCH_DEPTH];

        //load the vertices into shared memory
        #pragma unroll
        for(int i = 0; i < SHR_BATCH_DEPTH; i++){
            auto index = begin + i * BM + threadIdx.x;
            verticesS[0][i * BM + threadIdx.x] = index < end ? vertices[index] : make_float3(0, 0, 0);
        }

        //zero initialize the pixel coords
        #pragma unroll
        for(int i = 0; i < SHR_BATCH_DEPTH; i++){
            pixelCoords[i * BM + threadIdx.x] = make_int2(-1, -1);
        }

        __syncthreads();

        int stageIndex = 0;
        //loop over all vertexes assigned to this block
        #pragma unroll
        for(uint32 vi = begin + threadIdx.x; vi < end; vi += BM * SHR_BATCH_DEPTH){

            //load the next batch of vertices into shared memory
            #pragma unroll
            for(int i = 0; i < SHR_BATCH_DEPTH; i++){
                auto index = vi + i * BM;
                verticesS[(stageIndex + 1)%2][i * BM + threadIdx.x] = index < end ? vertices[index] : make_float3(0, 0, 0);
            }

            //break all out of bound threads to prevent repetitive conditional statements
            if(vi >= end) break;

            //compute the point of intersection of the ray with the image plane
            #pragma unroll
            for(int bi = 0; bi < SHR_BATCH_DEPTH; bi++){
                auto vertex = verticesS[stageIndex%2][bi * BM + threadIdx.x];

                //solve for the intersection parameters
                auto interParams = cramerIntersectSolver(cameraFrameR[0].positionE,
                                                         subtract3d(vertex, cameraFrameR[0].positionE),
                                                         cameraFrameR[0].imgTopLeft,
                                                         cameraFrameR[0].imgTopRight,
                                                         cameraFrameR[0].imgBottomLeft);

                //check if the intersection is within the image plane
                bool inRange = rectInRange(interParams.x, interParams.y);

                //exclude out of range intersections
                if(!inRange) {pixelCoords[bi * BM + threadIdx.x] = make_int2(-1, -1); continue;}

                //compute relative pixel coordinates
                int2 pixelCoord = make_int2((int)(interParams.x * cameraFrameR[0].resolutionX),
                                            (int)(interParams.y * cameraFrameR[0].resolutionY));

                //store the pixel coordinates
                pixelCoords[bi * BM + threadIdx.x] = pixelCoord;
            }

            __syncthreads();

            //store the pixel coordinates to global memory
            #pragma unroll
            for(auto bi = 0; bi < SHR_BATCH_DEPTH; bi++){
                if(pixelCoords[bi * BM + threadIdx.x].x >= 0 && pixelCoords[bi * BM + threadIdx.x].y >= 0) {
                    //R
                    imagePlane[pixelCoords[bi * BM + threadIdx.x].y * cameraFrameR[0].resolutionX +
                               pixelCoords[bi * BM + threadIdx.x].x] = 1;
                    //G
                    imagePlane[cameraFrameR[0].resolutionX * cameraFrameR[0].resolutionY * 1 +
                               pixelCoords[bi * BM + threadIdx.x].y * cameraFrameR[0].resolutionX +
                               pixelCoords[bi * BM + threadIdx.x].x] = 1;
                    //B
                    imagePlane[cameraFrameR[0].resolutionX * cameraFrameR[0].resolutionY * 2 +
                               pixelCoords[bi * BM + threadIdx.x].y * cameraFrameR[0].resolutionX +
                               pixelCoords[bi * BM + threadIdx.x].x] = 1;
                }
            }
        }
    }

    void renderVertices(float3 *vertices, uint32 numVertices, CameraFrame* cameraFrame, float* imagePlane,
                        int pixelCount){
        uint32 block = CUDA_BS_1D;
        uint32 grid = (numVertices + block - 1) / block;
        assertCudaError();

        cudaMemset(imagePlane, 0, pixelCount * sizeof(float) * 3);

        renderVerticesD<CUDA_BS_1D, 4><<<grid, block>>>(vertices, numVertices, cameraFrame, imagePlane);
        cudaDeviceSynchronize();
        assertCudaError();
    }



} // dylanrt