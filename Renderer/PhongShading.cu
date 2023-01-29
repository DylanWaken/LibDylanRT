//
// Created by dylan on 1/25/23.
//

#include "PhongShading.cuh"
#include "UtilFunctions.cuh"

#define AABB_PREFETCHED_LAYER_SIZE (1+ 2 + 4 + 8 + 16 + 32 + 64 + 128 + 256 + 512)
#define AABB_PREFETCHED_LAYERS 11
#define AABB_SEARCH_STACK_DEPTH 63

namespace dylanrt {

    struct solvedParams{
        bool solved;
        float3 trigParams;
        float3 trigV1;
        float3 trigV2;
        float3 trigV3;
        unsigned int trigIndex;
    };

    //we use slab method to check if a ray intersects with a box
    //basically we treat the box as pairs of parellel planes and gradually clip the ray
    //if the ray is still alive after clipping, then it intersects with the box
    __device__ __forceinline__ bool rayBoxIntersect(float3 max, float3 min, float3 e, float3 d){
        float t1x = (min.x - e.x) / d.x;
        float t2x = (max.x - e.x) / d.x;

        float t1y = (min.y - e.y) / d.y;
        float t2y = (max.y - e.y) / d.y;

        float t1z = (min.z - e.z) / d.z;
        float t2z = (max.z - e.z) / d.z;

        float tmin = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
        float tmax = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));

        if(tmax < 0) return false;
        return tmax >= tmin;
    }

    __device__ __forceinline__ solvedParams findIntersect(float3 e, float3 d, AABBnode* nodeShared, AABBnode* nodes,
                                                    triangle* trigs, float3* vertices){
        //register of node indices:
        unsigned int nodeIndices[AABB_SEARCH_STACK_DEPTH];
        int nodeIndicesTop = 0;
        //push root node (since the zero node stands for root)
        nodeIndices[0] = 0;
        nodeIndicesTop++;

        solvedParams result = {false, {0, 0, 1e20}, {0, 0, 0},
                               {0, 0, 0}, {0, 0, 0}, 0};
        #pragma unroll
        while(nodeIndicesTop > 0){
            //read the node correspond to the current stack top
            unsigned int nodeIndex = nodeIndices[nodeIndicesTop - 1];
            nodeIndicesTop--;

            //raea the node correspondent to the current stack top
            AABBnode node = nodeIndex < AABB_PREFETCHED_LAYER_SIZE ? nodeShared[nodeIndex] : nodes[nodeIndex];

            //if is leaf, return the triangle index
            if(node.isLeaf){
                float3 v1 = vertices[trigs[node.trigIndex].indices.x];
                float3 v2 = vertices[trigs[node.trigIndex].indices.y];
                float3 v3 = vertices[trigs[node.trigIndex].indices.z];
                float3 trigParams = cramerIntersectSolver(e, d, v1, v2, v3);
                if (trigInRange(trigParams.x, trigParams.y)){
                    if(abs(trigParams.z) <= abs(result.trigParams.z)){
                        result.solved = true;
                        result.trigParams = trigParams;
                        result.trigV1 = v1;
                        result.trigV2 = v2;
                        result.trigV3 = v3;
                        result.trigIndex = node.trigIndex;
                    }
                }
                continue;
            }

            //find the overlap between the left and right nodes of the current node
            AABBnode left = node.left < AABB_PREFETCHED_LAYER_SIZE ? nodeShared[node.left] : nodes[node.left];
            AABBnode right = node.right < AABB_PREFETCHED_LAYER_SIZE ? nodeShared[node.right] : nodes[node.right];

            bool rayIntersectLeft = rayBoxIntersect(left.maxPoint, left.minPoint, e, d);
            bool rayIntersectRight = rayBoxIntersect(right.maxPoint, right.minPoint, e, d);
            if(!rayIntersectLeft && !rayIntersectRight){
                continue;
            }

            //if the ray intersects with the left node, push it to the stack
            //put the closer node on the top of the stack to reduce repetitive search
            if(rayIntersectLeft){
                nodeIndices[nodeIndicesTop] = node.left;
                nodeIndicesTop++;
            }
            if(rayIntersectRight){
                nodeIndices[nodeIndicesTop] = node.right;
                nodeIndicesTop++;
            }
        }

        //asm("trap;");
        return result;
    }

    template<const int BM>
    __launch_bounds__(BM, 1)
    __global__ void phongShadingD(material* materials, triangle* trigs, float3* vertices, AABBnode* nodes, pointLight* lights,
                                  unsigned int numNodes, unsigned int numLights, CameraFrame* cameraFrame,float* imagePlane,
                                  unsigned int numPixls, float3 ambientLight){

        unsigned int tid = threadIdx.x;
        unsigned int bid = blockIdx.x;

        unsigned int begIndex = numPixls / gridDim.x * bid;
        unsigned int endIndex = gridDim.x == blockIdx.x + 1 ? numPixls : numPixls / gridDim.x * (bid + 1);

        //shared memory for the first layers:
        __shared__ AABBnode nodeShared[AABB_PREFETCHED_LAYER_SIZE];

        //read the camera frame
        CameraFrame frame = *cameraFrame;

        //copy range:
        unsigned int prefetchRange = min(numNodes, AABB_PREFETCHED_LAYER_SIZE);

        //all threads in the block prefetch the top layers
        //since these layers are used the most, we cache them in shared memory to improve the performance
        #pragma unroll
        for(auto i = tid; i < prefetchRange; i += BM){
            nodeShared[i] = nodes[i];
        }
        __syncthreads();

        //iterate through all pixels
        #pragma unroll
        for(auto i = begIndex + tid; i < endIndex; i += BM){
            //calculate pixel location
            float x = i%frame.resolutionX;
            float y = i/frame.resolutionX;

            //calculate the direction of the ray
            float picX = frame.imgTopLeft.x + (x) * ((frame.imgTopRight.x - frame.imgTopLeft.x)/frame.resolutionX) + (y) * ((frame.imgBottomLeft.x - frame.imgTopLeft.x)/frame.resolutionY);
            float picY = frame.imgTopLeft.y + (x) * ((frame.imgTopRight.y - frame.imgTopLeft.y)/frame.resolutionX) + (y) * ((frame.imgBottomLeft.y - frame.imgTopLeft.y)/frame.resolutionY);
            float picZ = frame.imgTopLeft.z + (x) * ((frame.imgTopRight.z - frame.imgTopLeft.z)/frame.resolutionX) + (y) * ((frame.imgBottomLeft.z - frame.imgTopLeft.z)/frame.resolutionY);

            //solve for ray direction
            float3 d = normalize3d(subtract3d(make_float3(picX, picY, picZ),frame.positionE));

            //find the closest triangle
            solvedParams trigParams = findIntersect(frame.positionE, d, nodeShared, nodes, trigs, vertices);

            float z0 = abs(norm3d(subtract3d(frame.positionE, make_float3(0,0,0))));

            if (trigParams.solved){
                imagePlane[i] = z0 - abs(trigParams.trigParams.z);
                imagePlane[numPixls*1 + i] = z0 - abs(trigParams.trigParams.z);
                imagePlane[numPixls*2 + i] = z0 - abs(trigParams.trigParams.z);
            }else{
                imagePlane[i] = 0.0f;
                imagePlane[numPixls*1 + i] = 0.0f;
                imagePlane[numPixls*2 + i] = 0.0f;
            }
        }
    }

    void phongShading(material* materials, triangle* trigs, float3* vertices, AABBnode* nodes, pointLight* lights,
                      unsigned int numNodes, unsigned int numLights, CameraFrame* cameraFrame,float* imagePlane,
                      unsigned int numPixls, float3 ambientLight){
        unsigned int blockSize = 256;
        unsigned int gridSize = (numPixls + blockSize - 1) / blockSize;
        phongShadingD<256><<<gridSize, blockSize>>>(materials, trigs, vertices, nodes,
                                                    lights, numNodes, numLights, cameraFrame, imagePlane,
                                                    numPixls, ambientLight);

        cudaDeviceSynchronize();
        assertCudaError();
    }
} // dylanrt