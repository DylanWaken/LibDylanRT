//
// Created by dylan on 1/22/23.
//

#include <cassert>
#include "UtilFunctions.cuh"
#define topOff(a,b) (((a)+(b) - 1)/(b))

namespace dylanrt {

    void* PUBLIC_REDUCTION_BUFFER;

    __host__ void assertCudaError(){
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA error");
        }
    }

    __device__ __forceinline__ float warpCompareM(float val) {
        #pragma unroll
        for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
            float temp = __shfl_xor_sync(0xffffffff, val, mask);
            val = temp >= val ? temp : val;
        }
        return val;
    }

    __device__ __forceinline__ float warpCompareS(float val) {
        #pragma unroll
        for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
            float temp = __shfl_xor_sync(0xffffffff, val, mask);
            val = temp <= val ? temp : val;
        }
        return val;
    }

    __host__ void reorientModel(TrigModel* model){
        for(int i = 0; i < model->numVertices; i++){
            auto x = model->vertices[i].x;
            auto y = model->vertices[i].y;
            auto z = model->vertices[i].z;

            model->vertices[i].x = x;
            model->vertices[i].y = z;
            model->vertices[i].z = -y;

            x = model->normals[i].x;
            y = model->normals[i].y;
            z = model->normals[i].z;

            model->normals[i].x = x;
            model->normals[i].y = z;
            model->normals[i].z = -y;
        }

        cudaMemcpy(model->verticesD, model->vertices, sizeof(float3) * model->numVertices, cudaMemcpyHostToDevice);
        cudaMemcpy(model->normalsD, model->normals, sizeof(float3) * model->numVertices, cudaMemcpyHostToDevice);
    }

    /**
     * Find the max value in the array A and store it in outA
     * This funtion need to be called recursively
     * @tparam BLOCK_WARPS
     * @param A
     * @param outA
     * @param reduceStep
     * @param procSize
     */
    template <const uint32 BLOCK_WARPS>
    __global__ void maxD(float* A, float* outA, uint32 reduceStep, uint32 procSize){
        uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32 reduceStepID = blockIdx.y;
        uint32 sourceStepID = blockIdx.z;
        uint32 tid = threadIdx.x;

        //warp reduction
        __shared__ float warpCache[BLOCK_WARPS];
        const uint32 warpId = tid / WARP_SIZE;
        const uint32 laneId = tid % WARP_SIZE;
        float max = idx < reduceStep && reduceStep * reduceStepID + idx < procSize
                    ? A[sourceStepID * procSize + reduceStepID * reduceStep + idx] : 0;
        __syncthreads();

        max = warpCompareM(max);
        if(laneId==0) warpCache[warpId] = max;

        __syncthreads();

        if(warpId==0){
            max = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            max = warpCompareM(max);
            if(laneId==0) outA[sourceStepID *
                               topOff(procSize, reduceStep) + reduceStepID] = max;
        }
    }

    template <const uint32 BLOCK_WARPS>
    __global__ void minD(float* A, float* outA, uint32 reduceStep, uint32 procSize){
        uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32 reduceStepID = blockIdx.y;
        uint32 sourceStepID = blockIdx.z;
        uint32 tid = threadIdx.x;

        //warp reduction
        __shared__ float warpCache[BLOCK_WARPS];
        const uint32 warpId = tid / WARP_SIZE;
        const uint32 laneId = tid % WARP_SIZE;
        float min = idx < reduceStep && reduceStep * reduceStepID + idx < procSize
                    ? A[sourceStepID * procSize + reduceStepID * reduceStep + idx] : 0;
        __syncthreads();

        min = warpCompareS(min);
        if(laneId==0) warpCache[warpId] = min;

        __syncthreads();

        if(warpId==0){
            min = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            min = warpCompareS(min);
            if(laneId==0) outA[sourceStepID *
                               topOff(procSize, reduceStep) + reduceStepID] = min;
        }
    }


    float maxVal(float* imagePlane, int resolutionX, int resolutionY){
        if(PUBLIC_REDUCTION_BUFFER == nullptr){
            cudaMalloc(&PUBLIC_REDUCTION_BUFFER, REDUCE_BUFFER_SIZE);
            cudaMemset(PUBLIC_REDUCTION_BUFFER, 0, REDUCE_BUFFER_SIZE);
        }

        uint32 procSize = resolutionX * resolutionY;

        float* src = imagePlane;
        float* buffer = (float*) PUBLIC_REDUCTION_BUFFER;

        while(procSize > 1){
            dim3 grid = dim3(1, topOff(procSize, REDUCE_BLOCK_SIZE), 1);
            uint32 block = REDUCE_BLOCK_SIZE;
            maxD<REDUCE_BLOCK_SIZE/WARP_SIZE><<<grid, block>>>
                 (src, buffer, REDUCE_BLOCK_SIZE, procSize);
            cudaDeviceSynchronize();
            assertCudaError();
            src = buffer;
            procSize = topOff(procSize, REDUCE_BLOCK_SIZE);
        }

        float max;
        cudaMemcpy(&max, buffer, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemset(PUBLIC_REDUCTION_BUFFER, 0, REDUCE_BUFFER_SIZE);
        return max;
    }

    float minVal(float* imagePlane, int resolutionX, int resolutionY){
        if(PUBLIC_REDUCTION_BUFFER == nullptr){
            cudaMalloc(&PUBLIC_REDUCTION_BUFFER, REDUCE_BUFFER_SIZE);
            cudaMemset(PUBLIC_REDUCTION_BUFFER, 0, REDUCE_BUFFER_SIZE);
        }

        uint32 procSize = resolutionX * resolutionY;

        float* src = imagePlane;
        float* buffer = (float*) PUBLIC_REDUCTION_BUFFER;

        while(procSize > 1){
            dim3 grid = dim3(1, topOff(procSize, REDUCE_BLOCK_SIZE), 1);
            uint32 block = REDUCE_BLOCK_SIZE;
            minD<REDUCE_BLOCK_SIZE/WARP_SIZE><<<grid, block>>>
                 (src, buffer, REDUCE_BLOCK_SIZE, procSize);
            cudaDeviceSynchronize();
            assertCudaError();
            src = buffer;
            procSize = topOff(procSize, REDUCE_BLOCK_SIZE);
        }

        float min;
        cudaMemcpy(&min, buffer, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemset(PUBLIC_REDUCTION_BUFFER, 0, REDUCE_BUFFER_SIZE);
        return min;
    }

    __global__ void normalizeImageD(const float* imagePlane, unsigned char* outPixels,
                                    int resolutionX, int resolutionY, float max, float min){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= resolutionY * resolutionX * 3) return;
        float val = imagePlane[idx];
        val = ((val - min) / (max - min)) * 255.0f;
        unsigned char out = (unsigned char) val;

        //NCHW to NHWC
        unsigned int c = idx / (resolutionX * resolutionY);
        unsigned int pos = idx % (resolutionX * resolutionY);
        unsigned int y = pos / resolutionX;
        unsigned int x = pos % resolutionX;

        outPixels[(y * resolutionX + x) * 3 + c] = out;


    }

    void createImage(float* imagePlane, unsigned char* outPixels, int resolutionX, int resolutionY){
        float max = maxVal(imagePlane, resolutionX, resolutionY);
        float min = minVal(imagePlane, resolutionX, resolutionY);
        unsigned int grid = topOff(resolutionX * resolutionY * 3, CUDA_BS_1D);
        uint32 block = CUDA_BS_1D;

        normalizeImageD<<<grid, block>>>(imagePlane, outPixels, resolutionX, resolutionY, max, min);
        cudaDeviceSynchronize();
        assertCudaError();
    }

} // dylanrt