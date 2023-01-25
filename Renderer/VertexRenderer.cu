//
// Created by dylan on 1/21/23.
//

#include <iostream>
#include <thread>
#include "VertexRenderer.cuh"
#include "chrono"

using namespace chrono;
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

        uint32 begin = (numVertices / (gridDim.x)) * blockIdx.x;
        uint32 end = blockIdx.x == gridDim.x - 1 ? numVertices : (numVertices / gridDim.x) * (blockIdx.x + 1);

        //store the pixel coords that are going to be illuminated
        __shared__ int2 pixelCoords[BM * SHR_BATCH_DEPTH];

        //store the vertices to process
        __shared__ float3 verticesS[2][BM * SHR_BATCH_DEPTH];

        float3 positionE = cameraFrame->positionE;
        int resolutionX = cameraFrame->resolutionX;
        int resolutionY = cameraFrame->resolutionY;
        float3 topLeft = cameraFrame->imgTopLeft;
        float3 topRight = cameraFrame->imgTopRight;
        float3 bottomLeft = cameraFrame->imgBottomLeft;

        //load the vertices into shared memory
        #pragma unroll SHR_BATCH_DEPTH
        for(int i = 0; i < SHR_BATCH_DEPTH; i++){
            auto index = begin + i * BM + threadIdx.x;
            verticesS[0][i * BM + threadIdx.x] = index < end ? vertices[index] : make_float3(0, 0, 0);
        }

        //zero initialize the pixel coords
        #pragma unroll SHR_BATCH_DEPTH
        for(int i = 0; i < SHR_BATCH_DEPTH; i++){
            pixelCoords[i * BM + threadIdx.x] = make_int2(-1, -1);
        }

        __syncthreads();

        int stageIndex = 0;
        //loop over all vertexes assigned to this block
        #pragma unroll
        for(uint32 vi = begin + threadIdx.x; vi < end; vi += BM * SHR_BATCH_DEPTH){

            //load the next batch of vertices into shared memory
            #pragma unroll SHR_BATCH_DEPTH
            for(int i = 0; i < SHR_BATCH_DEPTH; i++){
                auto index = vi + i * BM;
                verticesS[(stageIndex + 1)%2][i * BM + threadIdx.x] = index < end ? vertices[index] : make_float3(0, 0, 0);
            }

            //compute the point of intersection of the ray with the image plane
            #pragma unroll SHR_BATCH_DEPTH
            for(int bi = 0; bi < SHR_BATCH_DEPTH; bi++){
                auto vertex = verticesS[stageIndex%2][bi * BM + threadIdx.x];

                //solve for the intersection parameters
                auto interParams = cramerIntersectSolver(positionE,
                                                         subtract3d(vertex, positionE),
                                                         topLeft,
                                                         topRight,
                                                         bottomLeft);

                //check if the intersection is within the image plane
                bool inRange = rectInRange(interParams.x, interParams.y);

                //exclude out of range intersections
                if(!inRange) {pixelCoords[bi * BM + threadIdx.x] = make_int2(-1, -1); continue;}

                //compute relative pixel coordinates
                int2 pixelCoord = make_int2((int)(interParams.x * resolutionX),
                                            (int)(interParams.y * resolutionY));

                //store the pixel coordinates
                pixelCoords[bi * BM + threadIdx.x] = pixelCoord;
            }

            __syncthreads();

            //store the pixel coordinates to global memory
            #pragma unroll SHR_BATCH_DEPTH
            for(auto bi = 0; bi < SHR_BATCH_DEPTH; bi++){
                if(pixelCoords[bi * BM + threadIdx.x].x >= 0 && pixelCoords[bi * BM + threadIdx.x].y >= 0) {
                    //R
                    imagePlane[pixelCoords[bi * BM + threadIdx.x].y * resolutionX +
                               pixelCoords[bi * BM + threadIdx.x].x] = 1;
                    //G
                    imagePlane[resolutionX * resolutionY * 1 +
                               pixelCoords[bi * BM + threadIdx.x].y * resolutionX +
                               pixelCoords[bi * BM + threadIdx.x].x] = 1;
                    //B
                    imagePlane[resolutionX * resolutionY * 2 +
                               pixelCoords[bi * BM + threadIdx.x].y * resolutionX +
                               pixelCoords[bi * BM + threadIdx.x].x] = 1;
                }
            }
        }
    }

    #define BS 1
    void renderVertices(float3 *vertices, uint32 numVertices, CameraFrame* cameraFrame, float* imagePlane,
                        int pixelCount){


        uint32 block = CUDA_BS_1D;
        uint32 grid = (numVertices + (block*BS) - 1) / (block*BS);
        assertCudaError();

        cudaMemset(imagePlane, 0, pixelCount * sizeof(float) * 3);

        auto start = high_resolution_clock::now();

        renderVerticesD<CUDA_BS_1D, BS><<<grid, block>>>(vertices, numVertices, cameraFrame, imagePlane);
        cudaDeviceSynchronize();

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        std::cout << "renderVerticesD: " << duration.count() << std::endl;
        assertCudaError();
    }


    template<const int BM, const int BATCH_DEPTH>
    __global__ void renderEdgesD(float3 *vertices, triangle* trigs, unsigned int numTrigs,
                                 CameraFrame* cameraFrame, float* imagePlane,  int pixelCount){
        unsigned int begin = numTrigs / gridDim.x * blockIdx.x;
        unsigned int end = blockIdx.x == gridDim.x - 1 ? numTrigs : numTrigs / gridDim.x * (blockIdx.x + 1);

        //store cameraframe in registers
        CameraFrame cameraFrameR[1] = {*cameraFrame};

        //store the solved that are going to be illuminated
        __shared__ solvedTrig solvedTrigs[BM * BATCH_DEPTH];

        //store the pending vertices requested by each trig
        float3 verticesR[2][BATCH_DEPTH * 3];

        //load the first batch of vertices into register
        #pragma unroll
        for(int i = 0; i < BATCH_DEPTH; i++){
            auto index = begin + i * BM + threadIdx.x;
            if(index < end){
                verticesR[0][i * 3 + 0] = vertices[trigs[index].indices.x];
                verticesR[0][i * 3 + 1] = vertices[trigs[index].indices.y];
                verticesR[0][i * 3 + 2] = vertices[trigs[index].indices.z];
            } else {
                verticesR[0][i * 3 + 0] = make_float3(0, 0, 0);
                verticesR[0][i * 3 + 1] = make_float3(0, 0, 0);
                verticesR[0][i * 3 + 2] = make_float3(0, 0, 0);
            }
        }

        __syncthreads();

        int stageIndex = 0;
        #pragma unroll
        for(unsigned int ti = begin; ti < end; ti += BM * BATCH_DEPTH) {

            //load the next batch of vertices into register
            #pragma unroll
            for (int i = 0; i < BATCH_DEPTH; i++) {
                auto index = ti + i * BM + threadIdx.x;
                if (index < end) {
                    verticesR[(stageIndex + 1) % 2][i * 3 + 0] = vertices[trigs[index].indices.x];
                    verticesR[(stageIndex + 1) % 2][i * 3 + 1] = vertices[trigs[index].indices.y];
                    verticesR[(stageIndex + 1) % 2][i * 3 + 2] = vertices[trigs[index].indices.z];
                } else {
                    verticesR[(stageIndex + 1) % 2][i * 3 + 0] = make_float3(0, 0, 0);
                    verticesR[(stageIndex + 1) % 2][i * 3 + 1] = make_float3(0, 0, 0);
                    verticesR[(stageIndex + 1) % 2][i * 3 + 2] = make_float3(0, 0, 0);
                }
            }


            //break all out of bound threads to prevent repetitive conditional statements
            if (ti >= end) break;

            //solve the vertices in the current batch
            #pragma unroll
            for (int i = 0; i < BATCH_DEPTH; i++) {
                auto params1 = cramerIntersectSolver(cameraFrameR->positionE,
                                                     verticesR[stageIndex % 2][i * 3 + 0],
                                                     cameraFrameR[0].imgTopLeft,
                                                     cameraFrameR[0].imgTopRight,
                                                     cameraFrameR[0].imgBottomLeft);

                auto params2 = cramerIntersectSolver(cameraFrameR->positionE,
                                                     verticesR[stageIndex % 2][i * 3 + 1],
                                                     cameraFrameR[0].imgTopLeft,
                                                     cameraFrameR[0].imgTopRight,
                                                     cameraFrameR[0].imgBottomLeft);

                auto params3 = cramerIntersectSolver(cameraFrameR->positionE,
                                                     verticesR[stageIndex % 2][i * 3 + 2],
                                                     cameraFrameR[0].imgTopLeft,
                                                     cameraFrameR[0].imgTopRight,
                                                     cameraFrameR[0].imgBottomLeft);

                solvedTrigs[i * BM + threadIdx.x] = solvedTrig(params1, params2, params3);
            }

            __syncthreads();

            //map the solved edges to the image plane
            //Note: we draw a triangle when 1 of the 3 vertices is in the image plane
            #pragma unroll
            for (int i = 0; i < BATCH_DEPTH; i++) {
                bool inrange = solvedTrigs[i * BM + threadIdx.x];
            }
        }
    }


} // dylanrt