//
// Created by dylan on 1/25/23.
//

#include "PhongShading.cuh"
#include "UtilFunctions.cuh"
#include "chrono"

#define AABB_PREFETCHED_LAYER_SIZE (1+ 2 + 4 + 8 + 16 + 32 + 64 + 128 + 256 + 512)
#define AABB_PREFETCHED_LAYERS 11
#define AABB_SEARCH_STACK_DEPTH 24

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
    __device__ __forceinline__ bool rayBoxIntersect(float3 max, float3 min, float3 e, float3 inverseD){
        float t1x = (min.x - e.x) * inverseD.x;
        float t2x = (max.x - e.x) * inverseD.x;

        float t1y = (min.y - e.y) * inverseD.y;
        float t2y = (max.y - e.y) * inverseD.y;

        float t1z = (min.z - e.z) * inverseD.z;
        float t2z = (max.z - e.z) * inverseD.z;

        float tmin = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
        float tmax = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));

        if(tmax < 0) return false;
        return tmax >= tmin;
    }

    __device__ __forceinline__ bool rayBoxIntersect(float maxX, float maxY, float maxZ
                                                 , float minX, float minY, float minZ
                                                 , float eX, float eY, float eZ,
                                                 float invDX, float invDY, float invDZ){
        float t1x = (minX - eX) * invDX;
        float t2x = (maxX - eX) * invDX;

        float t1y = (minY - eY) * invDY;
        float t2y = (maxY - eY) * invDY;

        float t1z = (minZ - eZ) * invDZ;
        float t2z = (maxZ - eZ) * invDZ;

        float tmin = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
        float tmax = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));

        if(tmax < 0) return false;
        return tmax >= tmin;
    }

    __device__ __forceinline__ solvedParams findIntersect(float3 e, float3 d, float3 invD, AABBnode* nodeShared, AABBnode* nodes,
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

            bool rayIntersectLeft = rayBoxIntersect(left.maxPoint, left.minPoint, e, invD);
            bool rayIntersectRight = rayBoxIntersect(right.maxPoint, right.minPoint, e, invD);
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
    __launch_bounds__(BM, 2)
    __global__ void phongShadingDepthD(material* materials, triangle* trigs, float3* vertices, AABBnode* nodes, pointLight* lights,
                                       unsigned int numNodes, unsigned int numLights, CameraFrame* cameraFrame, float* imagePlane,
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
            float3 invD = make_float3(1.0f/d.x, 1.0f/d.y, 1.0f/d.z);

            //find the closest triangle
            solvedParams trigParams = findIntersect(frame.positionE, d, invD, nodeShared, nodes, trigs, vertices);

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

    template<const int PIXEL_PER_BLOCK>
    __global__ void phongShadingParaD(material* materials, triangle* trigs, float3* vertices, AABBnode* nodes, pointLight* lights,
                                      unsigned int numNodes, unsigned int numLights, CameraFrame* cameraFrame, float* imagePlane,
                                      unsigned int numPixls, float3 ambientLight){

        unsigned const int tid = threadIdx.x;
        unsigned const int bid = blockIdx.x;

        unsigned const int warpId = tid / 32;
        unsigned const int laneId = tid % 32;

        unsigned const int begIndex = numPixls / gridDim.x * bid;
        unsigned const int endIndex = gridDim.x == blockIdx.x + 1 ? numPixls : numPixls / gridDim.x * (bid + 1);

        unsigned const int BUFFER_SIZE = AABB_SEARCH_STACK_DEPTH * PIXEL_PER_BLOCK;

        __shared__ float directions[PIXEL_PER_BLOCK][3];
        __shared__ float invDirections[PIXEL_PER_BLOCK][3];
        __shared__ float solved[PIXEL_PER_BLOCK][3];

        //[pixel_ID ... | Node_access_request ...]
        extern __shared__ unsigned int stackData[];

        //shared memory for image plane, we can save registers by storing the image plane in shared memory
        __shared__ float imgTopLeft[3];
        __shared__ float imgTopRight[3];
        __shared__ float imgBottomLeft[3];
        __shared__ float eyePos[3];
        __shared__ unsigned int resolution[2];
        __shared__ int stackPtrPending[1];
        __shared__ int stackPtrProc[1];

        //read the camera frame
        if (tid == 0){
            imgTopRight[0] = cameraFrame->imgTopRight.x;
            imgTopRight[1] = cameraFrame->imgTopRight.y;
            imgTopRight[2] = cameraFrame->imgTopRight.z;

            imgTopLeft[0] = cameraFrame->imgTopLeft.x;
            imgTopLeft[1] = cameraFrame->imgTopLeft.y;
            imgTopLeft[2] = cameraFrame->imgTopLeft.z;

            imgBottomLeft[0] = cameraFrame->imgBottomLeft.x;
            imgBottomLeft[1] = cameraFrame->imgBottomLeft.y;
            imgBottomLeft[2] = cameraFrame->imgBottomLeft.z;

            eyePos[0] = cameraFrame->positionE.x;
            eyePos[1] = cameraFrame->positionE.y;
            eyePos[2] = cameraFrame->positionE.z;

            resolution[0] = cameraFrame->resolutionX;
            resolution[1] = cameraFrame->resolutionY;
        }
        __syncthreads();

        //start main loop
        #pragma unroll
        for (auto i = begIndex; i < endIndex; i+=PIXEL_PER_BLOCK){
            //precompute direction and invDirection
            if(tid < PIXEL_PER_BLOCK){
                //calculate pixel location
                float x = (i + tid)%resolution[0];
                float y = (i + tid)/resolution[0];

                //calculate the direction of the ray
                float picX = imgTopLeft[0] + (x) * ((imgTopRight[0] - imgTopLeft[0])/resolution[0]) + (y) * ((imgBottomLeft[0] - imgTopLeft[0])/resolution[1]);
                float picY = imgTopLeft[1] + (x) * ((imgTopRight[1] - imgTopLeft[1])/resolution[0]) + (y) * ((imgBottomLeft[1] - imgTopLeft[1])/resolution[1]);
                float picZ = imgTopLeft[2] + (x) * ((imgTopRight[2] - imgTopLeft[2])/resolution[0]) + (y) * ((imgBottomLeft[2] - imgTopLeft[2])/resolution[1]);

                //solve for ray direction
                float dX = picX - eyePos[0];
                float dY = picY - eyePos[1];
                float dZ = picZ - eyePos[2];

                directions[tid][0] = dX;
                directions[tid][1] = dY;
                directions[tid][2] = dZ;

                invDirections[tid][0] = 1.0f/dX;
                invDirections[tid][1] = 1.0f/dY;
                invDirections[tid][2] = 1.0f/dZ;

                solved[tid][0] = 0.0f;
                solved[tid][1] = 0.0f;
                solved[tid][2] = 0.0f;

                //assign pixel requests
                stackData[tid] = tid;
                //assign requests for the root node
                stackData[PIXEL_PER_BLOCK * AABB_SEARCH_STACK_DEPTH + tid] = 0;
                if(tid == 0) {
                    stackPtrPending[0] = PIXEL_PER_BLOCK;
                    stackPtrProc[0] = 0;
                }
            }

            __syncthreads();

            //start the first compute iteration
            int queued = 0;
            #pragma unroll
            while((queued = stackPtrPending[0] - stackPtrProc[0]) > 0){

                int readIndex = 0;
                //warp operation:
                if(laneId == 0){
                    readIndex = atomicAdd(stackPtrProc, min(queued, WARP_SIZE)) - min(queued, WARP_SIZE);
                }
                //get read index
                readIndex = __shfl_sync(0xffffffff, readIndex, 0) + (int)laneId;

                if(tid < queued) {

                    //compute the memory access index of this thread
                    readIndex = tid < queued ? (readIndex + tid) % BUFFER_SIZE : 0;

                    //get the pixel ID
                    unsigned int pixelID = stackData[readIndex];
                    //get the node ID
                    unsigned int nodeID = stackData[readIndex + BUFFER_SIZE];

                    //compute AABB intersection
                    bool intersect = rayBoxIntersect(nodes[nodeID].maxPoint.x, nodes[nodeID].maxPoint.y, nodes[nodeID].maxPoint.z,
                                                     nodes[nodeID].minPoint.x, nodes[nodeID].minPoint.y, nodes[nodeID].minPoint.z,
                                                     eyePos[0], eyePos[1], eyePos[2],
                                                     invDirections[pixelID][0], invDirections[pixelID][1], invDirections[pixelID][2]);

                    bool isLeaf = nodes[nodeID].isLeaf;
                    //is leaf
                    if(intersect && isLeaf){
                        //TODO : solve trig intersection...
                    }

                    //if the ray intersects, we add the children to the stack
                    int outputs = (intersect && !isLeaf) * 2;
                    int writeIndex = 0;

                    unsigned int leftChild = nodes[nodeID].left;
                    unsigned int rightChild = nodes[nodeID].right;

                    //reduce over all the outputs to get the write index and the total number of outputs
                    #pragma unroll
                    for(auto m = 0; m < tid; m++){
                        writeIndex += __shfl_sync(0xffffffff, outputs, m);
                    }

                    int writeIndexSrc = 0;

                    //cache write
                    //warp operation:
                    if((laneId == WARP_SIZE - 1 || laneId == queued - 1)){
                        writeIndexSrc = atomicAdd(stackPtrPending, writeIndex) - writeIndex;
                    }

                    if (intersect && !isLeaf){
                        //get write index
                        writeIndexSrc = __shfl_sync(0xffffffff, writeIndexSrc,
                                                    queued < WARP_SIZE ? queued - 1 : WARP_SIZE - 1);
                        //write the left child
                        stackData[writeIndexSrc + writeIndex - outputs] = pixelID;
                        stackData[writeIndexSrc + writeIndex - outputs + BUFFER_SIZE] = leftChild;
                        //write the right child
                        stackData[writeIndexSrc + writeIndex + 1 - outputs] = pixelID;
                        stackData[writeIndexSrc + writeIndex + 1 - outputs + BUFFER_SIZE] = rightChild;
                    }
                }
            }

            //more processing on the pixels ...


        }
    }

    void phongShading(material* materials, triangle* trigs, float3* vertices, AABBnode* nodes, pointLight* lights,
                      unsigned int numNodes, unsigned int numLights, CameraFrame* cameraFrame,float* imagePlane,
                      unsigned int numPixls, float3 ambientLight){
        unsigned int blockSize = 512;
        unsigned int gridSize = (numPixls/4 + blockSize - 1) / blockSize;
        auto t1 = std::chrono::high_resolution_clock::now();

        phongShadingDepthD < 512 ><<<gridSize, blockSize>>>(materials, trigs, vertices, nodes,
                lights, numNodes, numLights, cameraFrame, imagePlane,
                numPixls, ambientLight);

        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms" << std::endl;
        assertCudaError();
    }
} // dylanrt