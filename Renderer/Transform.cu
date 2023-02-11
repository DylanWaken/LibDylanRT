//
// Created by dylan on 2/7/23.
//

#include <chrono>
#include "Transform.cuh"


namespace dylanrt {

    //this method is optimized in CUDA that would reach theoritical maximum performance
    __global__ void
    transformD(float3 *vertices, float3 *screenVertices, float3 *screenSolved, bool *inrange, CameraFrame *cam) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

        //store the camera frame in registers
        float ax = cam->imgTopLeft.x;
        float ay = cam->imgTopLeft.y;
        float az = cam->imgTopLeft.z;

        float bx = cam->imgTopRight.x;
        float by = cam->imgTopRight.y;
        float bz = cam->imgTopRight.z;

        float cx = cam->imgBottomLeft.x;
        float cy = cam->imgBottomLeft.y;
        float cz = cam->imgBottomLeft.z;

        float ex = cam->positionE.x;
        float ey = cam->positionE.y;
        float ez = cam->positionE.z;

        float dx = vertices[idx].x - ex;
        float dy = vertices[idx].y - ey;
        float dz = vertices[idx].z - ez;

        //cramer solving
        float MInv = 1 / ((ax - bx) * ((ay - cy) * (dz) - (dy) * (az - cz))
                          + (ay - by) * ((dx) * (az - cz) - (ax - cx) * (dz))
                          + (az - bz) * ((ax - cx) * (dy) - (dx) * (ay - cy)));

        //first cramer determinant
        float beta = (ax - ex) * ((ay - cy) * (dz) - (az - cz) * (dy))
                     + (ay - ey) * ((az - cz) * (dx) - (ax - cx) * (dz))
                     + (az - ez) * ((ax - cx) * (dy) - (ay - cy) * (dx));

        beta *= MInv;

        //second cramer determinant
        float gamma = (ax - bx) * ((ay - ey) * (dz) - (az - ez) * (dy))
                      + (ay - by) * ((az - ez) * (dx) - (ax - ex) * (dz))
                      + (az - bz) * ((ax - ex) * (dy) - (ay - ey) * (dx));

        gamma *= MInv;

        //third cramer determinant
        float t = (ax - bx) * ((ay - cy) * (az - ez) - (az - cz) * (ay - ey))
                  + (ay - by) * ((az - cz) * (ax - ex) - (ax - cx) * (az - ez))
                  + (az - bz) * ((ax - cx) * (ay - ey) - (ay - cy) * (ax - ex));

        t *= MInv;

        //calculate the screen space vertex
        //beta
        float outx = (ax - beta * (bx - ax));
        float outy = (ay - beta * (by - ay));
        float outz = (az - beta * (bz - az));

        //gamma
        outx += (bx - gamma * (cx - ax));
        outy += (by - gamma * (cy - ay));
        outz += (bz - gamma * (cz - az));

        //store the screen space vertex
        screenVertices[idx].x = outx;
        screenVertices[idx].y = outy;
        screenVertices[idx].z = outz;
        screenSolved[idx].x = beta;
        screenSolved[idx].y = gamma;
        screenSolved[idx].z = t;
        inrange[idx] = (beta >= 0 && beta <= 1 && gamma >= 0 && gamma <= 1 && t >= 0);
    }

    void transform(float3 *vertices, float3 *screenVertices, float3 *screenSolved, CameraFrame *cam, bool *inrange,
                   unsigned int numVertices) {
        //calculate the number of blocks and threads
        unsigned int numThreads = 512;
        unsigned int numBlocks = (numVertices + numThreads - 1) / numThreads;

        //call the kernel
        transformD<<<numBlocks, numThreads>>>(vertices, screenVertices, screenSolved, inrange, cam);
        cudaDeviceSynchronize();

        assertCudaError();
    }

    template<const int BM>
    __global__ void frustumCullD(const bool *inrange, triangle* trigs, unsigned int numTrigs, unsigned int* outQueue) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ unsigned int queue[BM];
        __shared__ unsigned int heading[1];
        __shared__ unsigned int globalQueueIdx[1];

        if(idx >= numTrigs) return;

        //read the triangle of the thread
        triangle trig = trigs[idx];
        bool trigInrange = inrange[trig.indices.x] || inrange[trig.indices.y] || inrange[trig.indices.z];

        //if triangle in range, add to the queue, shared atomic add is much more efficient
        if (trigInrange) {
            unsigned int queueIdx = atomicAdd(heading, 1);
            queue[queueIdx] = idx;
        }

        //merge the queues into global output queue
        __syncthreads();

        if(threadIdx.x == 0) {
            globalQueueIdx[0] = atomicAdd(outQueue, heading[0]);
        }

        __syncthreads();

        if(threadIdx.x < heading[0]) {
            outQueue[globalQueueIdx[0] + threadIdx.x +1] = queue[threadIdx.x];
        }
    }

    //we subdivide each frame into a grid of smaller bins
    //each block would contain the label of triangle that is in the block
    //this is a block function that can be executed by a block
    template<unsigned int BATCH_SIZE>
    __global__ void binRasterD(unsigned int* inQueue, unsigned int numTrigs, triangle *trigs, float3 *screenSolved,
                              unsigned int binX, unsigned int binY, float binStepX, float binStepY, unsigned int *binTrigQueues) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

        //[ ... Queue Header ... | ... Queue Writeback Header ... | ... Queue Data ... ]
        extern __shared__ unsigned int outQueues[];

        //queue header counts the number of triangles in the queue
        //size is binX * binY
        unsigned int *queueHeader = outQueues;
        unsigned int *queueWritebackHeader = outQueues + binX * binY;
        unsigned int toProc = inQueue[0];

        //queue data stores the triangle index
        //size is BATCH_SIZE * binX * binY
        unsigned int *queueData = outQueues + 2 * binX * binY;

        //zero initialize the queue header
        #pragma unroll
        for (auto i = threadIdx.x; i < binX * binY; i += blockDim.x) {
            queueHeader[i] = 0;
        }

        if (idx >= toProc) return;

        //wait for all threads to finish
        __syncthreads();

        //read the triangle of the thread
        triangle trig = trigs[inQueue[idx+1]];

        //read the screen space vertex
        float x0 = screenSolved[trig.indices.x].x;
        float y0 = screenSolved[trig.indices.x].y;
        float x1 = screenSolved[trig.indices.y].x;
        float y1 = screenSolved[trig.indices.y].y;
        float x2 = screenSolved[trig.indices.z].x;
        float y2 = screenSolved[trig.indices.z].y;

        //calculate the bounding box of the triangle
        float minX = fminf(fminf(x0, x1), x2);
        float minY = fminf(fminf(y0, y1), y2);
        float maxX = fmaxf(fmaxf(x0, x1), x2);
        float maxY = fmaxf(fmaxf(y0, y1), y2);

        //calculate the bin index of the bounding b
        unsigned int binMinX = floor(minX / binStepX);
        unsigned int binMinY = floor(minY / binStepY);
        unsigned int binMaxX = ceil(maxX / binStepX);
        unsigned int binMaxY = ceil(maxY / binStepY);

        //clamp the bin index
        binMinX = binMinX < binX ? binMinX : binX - 1;
        binMinY = binMinY < binY ? binMinY : binY - 1;
        binMaxX = binMaxX < binX ? binMaxX : binX - 1;
        binMaxY = binMaxY < binY ? binMaxY : binY - 1;

        //for each bin in the bounding box
        #pragma unroll
        for(unsigned int i = binMinX; i <= binMaxX; i++){
            #pragma unroll
            for(unsigned int j = binMinY; j <= binMaxY; j++){
                //calculate the queue index
                unsigned int queueIdx = i * binY + j;

                //increment the queue header, we reserve this slot for our triangle
                unsigned int out = atomicAdd(&queueHeader[queueIdx],1);

                //calculate the queue data index
                unsigned int queueDataIdx = queueIdx * BATCH_SIZE + (out);

                //store the triangle index in the queue
                queueData[queueDataIdx] = idx;
            }
        }

        //wait for all threads to finish
        __syncthreads();

        //reserve space in the global queue
        #pragma unroll
        for(auto i = threadIdx.x; i < binX * binY; i+= blockDim.x){
            queueWritebackHeader[i] = atomicAdd(&binTrigQueues[i], queueHeader[i]);
        }

        //wait for all threads to finish
        __syncthreads();

        //find the write back index for this thread
        unsigned int writebackIdx = threadIdx.x;
        unsigned int writebackQueueIdx = 0;
        #pragma unroll
        while(writebackIdx >= queueHeader[writebackQueueIdx]){
            writebackIdx -= queueHeader[writebackQueueIdx];
            writebackQueueIdx++;
        }

        //write back the triangle index
        binTrigQueues[writebackQueueIdx * numTrigs + queueWritebackHeader[writebackQueueIdx] + writebackIdx]
                    = queueData[writebackQueueIdx * BATCH_SIZE + writebackIdx];
    }

    void frustumCull(const bool *inrange, triangle* trigs, unsigned int numTrigs, unsigned int* outQueue) {
        //calculate the number of blocks and threads
        unsigned int numThreads = 512;
        unsigned int numBlocks = (numTrigs + numThreads - 1) / numThreads;

        //do the frustum culling
        frustumCullD<512><<<numBlocks, numThreads>>>(inrange, trigs, numTrigs, outQueue);
        cudaDeviceSynchronize();
        assertCudaError();
    }

    void binRasterize(triangle* trigs, unsigned int numTrigs, float3* screenSolved, unsigned int* inQueue, unsigned int* outQueue, unsigned int binX, unsigned int binY, float binStepX, float binStepY) {
        //calculate the number of blocks and threads
        unsigned int numThreads = 512;
        unsigned int numBlocks = (numTrigs + numThreads - 1) / numThreads;

        //do the bin rasterization
        binRasterD<512><<<numBlocks, numThreads, 2 * binX * binY * sizeof(unsigned int) +
                                                 512 * binX * binY * sizeof(unsigned int)>>>
                (inQueue, numTrigs, trigs, screenSolved, binX, binY, binStepX, binStepY, outQueue);
        cudaDeviceSynchronize();
        assertCudaError();
    }
}