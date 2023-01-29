//
// Created by dylan on 1/21/23.
//

#include <iostream>
#include <thread>
#include "VertexRenderer.cuh"
#include "chrono"

using namespace chrono;
namespace dylanrt {


    template<const int BM, const int SHR_BATCH_DEPTH>
    __global__ void renderVerticesD(float3 *vertices, uint32 numVertices,
                                    CameraFrame* cameraFrame, float* imagePlane){

        uint32 begin = (numVertices / (gridDim.x)) * blockIdx.x;
        uint32 end = blockIdx.x == gridDim.x - 1 ? numVertices : (numVertices / gridDim.x) * (blockIdx.x + 1);

        //store the pixel coords that are going to be illuminated
        __shared__ int2 pixelCoords[BM * SHR_BATCH_DEPTH];

        //store the vertices in registers
        float3 verticesS[2][SHR_BATCH_DEPTH];

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
            verticesS[0][i] = index < end ? vertices[index] : make_float3(0, 0, 0);
        }

        //zero initialize the pixel coords
        #pragma unroll SHR_BATCH_DEPTH
        for(int i = 0; i < SHR_BATCH_DEPTH; i++){
            pixelCoords[i] = make_int2(-1, -1);
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
                verticesS[(stageIndex + 1)%2][i] = index < end ? vertices[index] : make_float3(0, 0, 0);
            }

            //compute the point of intersection of the ray with the image plane
            #pragma unroll SHR_BATCH_DEPTH
            for(int bi = 0; bi < SHR_BATCH_DEPTH; bi++){
                auto vertex = verticesS[stageIndex%2][bi];

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

    #define BS 4
    void renderVertices(float3 *vertices, uint32 numVertices, CameraFrame* cameraFrame, float* imagePlane,
                        int pixelCount){


        uint32 block = CUDA_BS_1D;
        uint32 grid = (numVertices + (block*BS) - 1) / (block*BS);
        assertCudaError();

        cudaMemset(imagePlane, 0, pixelCount * sizeof(float) * 3);

        renderVerticesD<CUDA_BS_1D, BS><<<grid, block>>>(vertices, numVertices, cameraFrame, imagePlane);
        cudaDeviceSynchronize();
        assertCudaError();
    }

    //we would only use the alpha beta values for sampling the edges
    __device__ __forceinline__ void sampleEdges(float3 solvedPt1, float3 solvedPt2, float* imagePlane,
                                                 int resolutionX, int resolutionY){
        auto stepLengthX = 1.0f / resolutionX;
        auto stepLengthY = 1.0f / resolutionY;

        auto steplength = stepLengthX < stepLengthY ? stepLengthX : stepLengthY;
        auto xStep = steplength * (solvedPt2.x - solvedPt1.x) / sqrtf((solvedPt2.x - solvedPt1.x) * (solvedPt2.x - solvedPt1.x) +
                           (solvedPt2.y - solvedPt1.y) * (solvedPt2.y - solvedPt1.y));

        auto yStep = steplength * (solvedPt2.y - solvedPt1.y)/ sqrtf((solvedPt2.x - solvedPt1.x) * (solvedPt2.x - solvedPt1.x) +
                           (solvedPt2.y - solvedPt1.y) * (solvedPt2.y - solvedPt1.y));

        float x0 = solvedPt1.x;
        float y0 = solvedPt1.y;

        #pragma unroll
        while(sqrt((x0 - solvedPt2.x)*(x0 - solvedPt2.x) + (y0 - solvedPt2.y)*(y0 - solvedPt2.y
        )) >= steplength && x0 > 0 && y0 > 0 && x0 < 1 && y0 < 1){

            //R
            imagePlane[(int)(y0 * resolutionY) * resolutionX + (int)(x0 * resolutionX)] = 1;
            //G
            imagePlane[resolutionX * resolutionY * 1 +
                       (int)(y0 * resolutionY) * resolutionX + (int)(x0 * resolutionX)] = 1;
            //B
            imagePlane[resolutionX * resolutionY * 2 +
                       (int)(y0 * resolutionY) * resolutionX + (int)(x0 * resolutionX)] = 1;

            x0 += xStep;
            y0 += yStep;
        }
    }


    template<const int BM, const int BATCH_DEPTH>
    __global__ void renderEdgesD(float3 *vertices, triangle* trigs, unsigned int numTrigs,
                                 CameraFrame* cameraFrame, float* imagePlane){
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
                                                     subtract3d(verticesR[stageIndex % 2][i * 3 + 0], cameraFrameR->positionE),
                                                     cameraFrameR[0].imgTopLeft,
                                                     cameraFrameR[0].imgTopRight,
                                                     cameraFrameR[0].imgBottomLeft);

                auto params2 = cramerIntersectSolver(cameraFrameR->positionE,
                                                     subtract3d(verticesR[stageIndex % 2][i * 3 + 1], cameraFrameR->positionE),
                                                     cameraFrameR[0].imgTopLeft,
                                                     cameraFrameR[0].imgTopRight,
                                                     cameraFrameR[0].imgBottomLeft);

                auto params3 = cramerIntersectSolver(cameraFrameR->positionE,
                                                     subtract3d(verticesR[stageIndex % 2][i * 3 + 2], cameraFrameR->positionE),
                                                     cameraFrameR[0].imgTopLeft,
                                                     cameraFrameR[0].imgTopRight,
                                                     cameraFrameR[0].imgBottomLeft);

                solvedTrigs[i * BM + threadIdx.x] = {params1, params2, params3};
            }

            __syncthreads();

            //map the solved edges to the image plane
            //Note: we draw a triangle when 1 of the 3 vertices is in the image plane
            #pragma unroll
            for (int i = 0; i < BATCH_DEPTH; i++) {
                 auto point1 = solvedTrigs[i * BM + threadIdx.x].scrPoint1;
                 auto point2 = solvedTrigs[i * BM + threadIdx.x].scrPoint2;
                 auto point3 = solvedTrigs[i * BM + threadIdx.x].scrPoint3;

                 bool inRange = rectInRange(point1.x, point1.y) ||
                                rectInRange(point2.x, point2.y) ||
                                rectInRange(point3.x, point3.y);

                 if(!inRange) continue;

                 //sample pixel cords according to the line connecting the 2 points
                sampleEdges(point1,point2, imagePlane, cameraFrameR->resolutionX, cameraFrameR->resolutionY);
                sampleEdges(point2,point3, imagePlane, cameraFrameR->resolutionX, cameraFrameR->resolutionY);
                sampleEdges(point3,point1, imagePlane, cameraFrameR->resolutionX, cameraFrameR->resolutionY);
            }
        }
    }

    #define BS2 4
    void renderEdges(float3 *vertices, triangle* trigs, unsigned int numTrigs,
                     CameraFrame* cameraFrame, float* imagePlane,  int pixelCount){


        uint32 block = CUDA_BS_1D;
        uint32 grid = (numTrigs + (block*BS2) - 1) / (block*BS2);
        cudaMemset(imagePlane, 0, pixelCount * sizeof(float) * 3);
        assertCudaError();

        renderEdgesD<CUDA_BS_1D, BS2><<<grid, block>>>(vertices,trigs, numTrigs, cameraFrame,imagePlane);
        cudaDeviceSynchronize();
        assertCudaError();
    }

} // dylanrt