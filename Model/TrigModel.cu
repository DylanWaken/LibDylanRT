//
// Created by dylan on 1/20/23.
//

#include <float.h>
#include <algorithm>
#include "TrigModel.cuh"
#include "../Renderer/UtilFunctions.cuh"
#include "TinyGLTF/tiny_gltf.h"
#include <queue>

using namespace tinygltf;
namespace dylanrt {

    __host__ float objMedianSplit(vector<uint32_t>* indices, triangle* trigs, float3* vertices, int dim, uint32_t beg, uint32_t end){
        if(dim == 0){
            //find the objective median
            float objMedian = (vertices[trigs[(*indices)[(end+beg) / 2]].indices.x].x
                               + vertices[trigs[(*indices)[(end+beg) / 2]].indices.y].x
                               + vertices[trigs[(*indices)[(end+beg) / 2]].indices.z].x) / 3.0f;

            return objMedian;
        }

        if(dim == 1){
            //find the objective median
            float objMedian = (vertices[trigs[(*indices)[(end+beg) / 2]].indices.x].y
                               + vertices[trigs[(*indices)[(end+beg) / 2]].indices.y].y
                               + vertices[trigs[(*indices)[(end+beg) / 2]].indices.z].y) / 3.0f;

            return objMedian;
        }

        if(dim == 2){
            //find the objective median
            float objMedian = (vertices[trigs[(*indices)[(end+beg) / 2]].indices.x].z
                               + vertices[trigs[(*indices)[(end+beg) / 2]].indices.y].z
                               + vertices[trigs[(*indices)[(end+beg) / 2]].indices.z].z) / 3.0f;

            return objMedian;
        }
    }

    __host__ float3 getTrigCenter(triangle& trig, float3* vertices){
        float centerX = (vertices[trig.indices.x].x + vertices[trig.indices.y].x + vertices[trig.indices.z].x) / 3.0f;
        float centerY = (vertices[trig.indices.x].y + vertices[trig.indices.y].y + vertices[trig.indices.z].y) / 3.0f;
        float centerZ = (vertices[trig.indices.x].z + vertices[trig.indices.y].z + vertices[trig.indices.z].z) / 3.0f;
        return make_float3(centerX, centerY, centerZ);
    }

    __host__ float3 getTrigMin(triangle& trig, float3* vertices){
        float minX = min(min(vertices[trig.indices.x].x, vertices[trig.indices.y].x), vertices[trig.indices.z].x);
        float minY = min(min(vertices[trig.indices.x].y, vertices[trig.indices.y].y), vertices[trig.indices.z].y);
        float minZ = min(min(vertices[trig.indices.x].z, vertices[trig.indices.y].z), vertices[trig.indices.z].z);
        return make_float3(minX, minY, minZ);
    }

    __host__ float3 getTrigMax(triangle& trig, float3* vertices){
        float maxX = max(max(vertices[trig.indices.x].x, vertices[trig.indices.y].x), vertices[trig.indices.z].x);
        float maxY = max(max(vertices[trig.indices.x].y, vertices[trig.indices.y].y), vertices[trig.indices.z].y);
        float maxZ = max(max(vertices[trig.indices.x].z, vertices[trig.indices.y].z), vertices[trig.indices.z].z);
        return make_float3(maxX, maxY, maxZ);
    }


    struct AABBNodeTemp{
        AABBNodeTemp* left{};
        AABBNodeTemp* right{};
        AABBNodeTemp* parent{};
        vector<uint32_t>* indices;
        triangle* trigs;
        float3* vertices;
        int* numNodes;

        uint32_t indBegin = 0;
        uint32_t indEnd = 0;

        float3 minPoint;
        float3 maxPoint;

        //if leaf
        bool isLeaf = false;
        uint32_t trigIndex{};

        AABBNodeTemp(float3 max, float3 min, triangle* trig, float3* vertices, vector<uint32_t>* indices, int* numNodes)
                : maxPoint(max), minPoint(min), trigs(trig), vertices(vertices), numNodes(numNodes), indices(indices){}

        void build(){
            assert(indEnd > indBegin);

            if(indEnd - indBegin == 1){
                isLeaf = true;
                trigIndex = (*indices)[indBegin];
            }

            else {

                isLeaf = false;
                //find the longest dimension
                float3 dim = subtract3d(maxPoint, minPoint);
                int longestDim = 0;
                float dimLength = dim.x;
                if (dim.y > dimLength) {
                    longestDim = 1;
                }
                if (dim.z > dimLength) {
                    longestDim = 2;
                }

                //partial sort the indices between indBegin and indEnd
                if (longestDim == 0) {
                    sort(indices->begin() + indBegin, indices->begin() + indEnd + 1,
                         [this](uint32_t a, uint32_t b) {
                             return (vertices[trigs[a].indices.x].x + vertices[trigs[a].indices.y].x + vertices[trigs[a].indices.z].x) / 3.0f
                                    < (vertices[trigs[b].indices.x].x + vertices[trigs[b].indices.y].x + vertices[trigs[b].indices.z].x) / 3.0f;
                         });
                }

                if (longestDim == 1) {
                    sort(indices->begin() + indBegin, indices->begin() + indEnd + 1,
                         [this](uint32_t a, uint32_t b) {
                             return (vertices[trigs[a].indices.x].y + vertices[trigs[a].indices.y].y + vertices[trigs[a].indices.z].y) / 3.0f
                                    < (vertices[trigs[b].indices.x].y + vertices[trigs[b].indices.y].y + vertices[trigs[b].indices.z].y) / 3.0f;
                         });
                }

                if (longestDim == 2) {
                    sort(indices->begin() + indBegin, indices->begin() + indEnd + 1,
                         [this](uint32_t a, uint32_t b) {
                             return (vertices[trigs[a].indices.x].z + vertices[trigs[a].indices.y].z + vertices[trigs[a].indices.z].z) / 3.0f
                                    < (vertices[trigs[b].indices.x].z + vertices[trigs[b].indices.y].z + vertices[trigs[b].indices.z].z) / 3.0f;
                         });
                }

                //find the objective median
                float objMedian = objMedianSplit(indices, trigs, vertices, longestDim, indBegin, indEnd);

                //recompute the maxPoint minPoint for both sub nodes
                if (longestDim == 0) {
                    float3 leftMax = make_float3(objMedian, maxPoint.y, maxPoint.z);
                    float3 leftMin = make_float3(minPoint.x, minPoint.y, minPoint.z);

                    float3 rightMax = make_float3(maxPoint.x, maxPoint.y, maxPoint.z);
                    float3 rightMin = make_float3(objMedian, minPoint.y, minPoint.z);

                    cudaMallocHost(&left, sizeof(AABBNodeTemp));
                    cudaMallocHost(&right, sizeof(AABBNodeTemp));

                    left->maxPoint = leftMax;
                    left->minPoint = leftMin;
                    left->trigs = trigs;
                    left->vertices = vertices;
                    left->numNodes = numNodes;
                    left->indices = indices;

                    right->maxPoint = rightMax;
                    right->minPoint = rightMin;
                    right->trigs = trigs;
                    right->vertices = vertices;
                    right->numNodes = numNodes;
                    right->indices = indices;

                    left->parent = this;
                    right->parent = this;

                    uint32_t boarderInd = indBegin;

                    //split the indices
                    for (auto i = indBegin; i < indEnd; ++i) {
                        float3 trigCenter = getTrigCenter(trigs[(*indices)[i]], vertices);
                        if ((trigCenter.x < objMedian || boarderInd == indBegin)&&(boarderInd < indEnd-1)) {
                            boarderInd++;
                        }
                    }

                    //calibrate minPoint maxPoint of the children
                    for (auto i = indBegin; i < boarderInd; ++i) {
                        float3 trigMin = getTrigMin(trigs[(*indices)[i]], vertices);
                        float3 trigMax = getTrigMax(trigs[(*indices)[i]], vertices);

                        left->minPoint.x = min(left->minPoint.x, trigMin.x);
                        left->minPoint.y = min(left->minPoint.y, trigMin.y);
                        left->minPoint.z = min(left->minPoint.z, trigMin.z);

                        left->maxPoint.x = max(left->maxPoint.x, trigMax.x);
                        left->maxPoint.y = max(left->maxPoint.y, trigMax.y);
                        left->maxPoint.z = max(left->maxPoint.z, trigMax.z);
                    }

                    for (auto i = boarderInd; i < indEnd; ++i) {
                        float3 trigMin = getTrigMin(trigs[(*indices)[i]], vertices);
                        float3 trigMax = getTrigMax(trigs[(*indices)[i]], vertices);

                        right->minPoint.x = min(right->minPoint.x, trigMin.x);
                        right->minPoint.y = min(right->minPoint.y, trigMin.y);
                        right->minPoint.z = min(right->minPoint.z, trigMin.z);

                        right->maxPoint.x = max(right->maxPoint.x, trigMax.x);
                        right->maxPoint.y = max(right->maxPoint.y, trigMax.y);
                        right->maxPoint.z = max(right->maxPoint.z, trigMax.z);
                    }

                    left->indBegin = indBegin;
                    left->indEnd = boarderInd;
                    right->indBegin = boarderInd;
                    right->indEnd = indEnd;

                    numNodes[0] += 2;

                    //build the children
                    left->build();
                    right->build();
                }

                if (longestDim == 1) {
                    float3 leftMax = make_float3(maxPoint.x, objMedian, maxPoint.z);
                    float3 leftMin = make_float3(minPoint.x, minPoint.y, minPoint.z);

                    float3 rightMax = make_float3(maxPoint.x, maxPoint.y, maxPoint.z);
                    float3 rightMin = make_float3(minPoint.x, objMedian, minPoint.z);

                    cudaMallocHost(&left, sizeof(AABBNodeTemp));
                    cudaMallocHost(&right, sizeof(AABBNodeTemp));

                    left->maxPoint = leftMax;
                    left->minPoint = leftMin;
                    left->trigs = trigs;
                    left->vertices = vertices;
                    left->numNodes = numNodes;
                    left->indices = indices;

                    right->maxPoint = rightMax;
                    right->minPoint = rightMin;
                    right->trigs = trigs;
                    right->vertices = vertices;
                    right->numNodes = numNodes;
                    right->indices = indices;

                    left->parent = this;
                    right->parent = this;

                    uint32_t boarderInd = indBegin;

                    //split the indices
                    for (auto i = indBegin; i < indEnd; ++i) {
                        float3 trigCenter = getTrigCenter(trigs[(*indices)[i]], vertices);
                        if ((trigCenter.y < objMedian || boarderInd == indBegin) && (boarderInd < indEnd - 1)) {
                            boarderInd++;
                        }
                    }

                    //calibrate minPoint maxPoint of the children
                    for (auto i = indBegin; i < boarderInd; ++i) {
                        float3 trigMin = getTrigMin(trigs[(*indices)[i]], vertices);
                        float3 trigMax = getTrigMax(trigs[(*indices)[i]], vertices);

                        left->minPoint.x = min(left->minPoint.x, trigMin.x);
                        left->minPoint.y = min(left->minPoint.y, trigMin.y);
                        left->minPoint.z = min(left->minPoint.z, trigMin.z);

                        left->maxPoint.x = max(left->maxPoint.x, trigMax.x);
                        left->maxPoint.y = max(left->maxPoint.y, trigMax.y);
                        left->maxPoint.z = max(left->maxPoint.z, trigMax.z);
                    }

                    for (auto i = boarderInd; i < indEnd; ++i) {
                        float3 trigMin = getTrigMin(trigs[(*indices)[i]], vertices);
                        float3 trigMax = getTrigMax(trigs[(*indices)[i]], vertices);

                        right->minPoint.x = min(right->minPoint.x, trigMin.x);
                        right->minPoint.y = min(right->minPoint.y, trigMin.y);
                        right->minPoint.z = min(right->minPoint.z, trigMin.z);

                        right->maxPoint.x = max(right->maxPoint.x, trigMax.x);
                        right->maxPoint.y = max(right->maxPoint.y, trigMax.y);
                        right->maxPoint.z = max(right->maxPoint.z, trigMax.z);
                    }

                    left->indBegin = indBegin;
                    left->indEnd = boarderInd;
                    right->indBegin = boarderInd;
                    right->indEnd = indEnd;

                    numNodes[0] += 2;

                    //build the children
                    left->build();
                    right->build();
                }

                if (longestDim == 2) {
                    float3 leftMax = make_float3(maxPoint.x, maxPoint.y, objMedian);
                    float3 leftMin = make_float3(minPoint.x, minPoint.y, minPoint.z);

                    float3 rightMax = make_float3(maxPoint.x, maxPoint.y, maxPoint.z);
                    float3 rightMin = make_float3(minPoint.x, minPoint.y, objMedian);

                    cudaMallocHost(&left, sizeof(AABBNodeTemp));
                    cudaMallocHost(&right, sizeof(AABBNodeTemp));

                    left->maxPoint = leftMax;
                    left->minPoint = leftMin;
                    left->trigs = trigs;
                    left->vertices = vertices;
                    left->numNodes = numNodes;
                    left->indices = indices;

                    right->maxPoint = rightMax;
                    right->minPoint = rightMin;
                    right->trigs = trigs;
                    right->vertices = vertices;
                    right->numNodes = numNodes;
                    right->indices = indices;

                    left->parent = this;
                    right->parent = this;

                    uint32_t boarderInd = indBegin;

                    //split the indices
                    for (auto i = indBegin; i < indEnd; ++i) {
                        float3 trigCenter = getTrigCenter(trigs[(*indices)[i]], vertices);
                        if ((trigCenter.z < objMedian || boarderInd == indBegin) && (boarderInd < indEnd - 1)) {
                            boarderInd++;
                        }
                    }

                    //calibrate minPoint maxPoint of the children
                    for (auto i = indBegin; i < boarderInd; ++i) {
                        float3 trigMin = getTrigMin(trigs[(*indices)[i]], vertices);
                        float3 trigMax = getTrigMax(trigs[(*indices)[i]], vertices);

                        left->minPoint.x = min(left->minPoint.x, trigMin.x);
                        left->minPoint.y = min(left->minPoint.y, trigMin.y);
                        left->minPoint.z = min(left->minPoint.z, trigMin.z);

                        left->maxPoint.x = max(left->maxPoint.x, trigMax.x);
                        left->maxPoint.y = max(left->maxPoint.y, trigMax.y);
                        left->maxPoint.z = max(left->maxPoint.z, trigMax.z);
                    }

                    for (auto i = boarderInd; i < indEnd; ++i) {
                        float3 trigMin = getTrigMin(trigs[(*indices)[i]], vertices);
                        float3 trigMax = getTrigMax(trigs[(*indices)[i]], vertices);

                        right->minPoint.x = min(right->minPoint.x, trigMin.x);
                        right->minPoint.y = min(right->minPoint.y, trigMin.y);
                        right->minPoint.z = min(right->minPoint.z, trigMin.z);

                        right->maxPoint.x = max(right->maxPoint.x, trigMax.x);
                        right->maxPoint.y = max(right->maxPoint.y, trigMax.y);
                        right->maxPoint.z = max(right->maxPoint.z, trigMax.z);
                    }

                    left->indBegin = indBegin;
                    left->indEnd = boarderInd;
                    right->indBegin = boarderInd;
                    right->indEnd = indEnd;

                    numNodes[0] += 2;

                    //build the children
                    left->build();
                    right->build();
                }
            }
        }
    };

    void buildAABBTree(AABBTree& tree, triangle* trigs, uint32_t numTrigs,
                       float3* vertices, uint32_t numVertices){

        //initialize the array of trig indices
        vector<uint32_t> trigIndices(numTrigs);
        for (int i = 0; i < numTrigs; ++i) {
            trigIndices[i] = i;
        }

        //find the minPoint maxPoint of the model
        float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (int i = 0; i < numVertices; ++i) {
            min.x = min.x < vertices[i].x ? min.x : vertices[i].x;
            min.y = min.y < vertices[i].y ? min.y : vertices[i].y;
            min.z = min.z < vertices[i].z ? min.z : vertices[i].z;

            max.x = max.x > vertices[i].x ? max.x : vertices[i].x;
            max.y = max.y > vertices[i].y ? max.y : vertices[i].y;
            max.z = max.z > vertices[i].z ? max.z : vertices[i].z;
        }

        int numNodes = 0;
        auto root = new AABBNodeTemp(max, min, trigs, vertices, &trigIndices, &numNodes);
        root->indBegin = 0;
        root->indEnd = numTrigs;

        unsigned int count = 0;

        //This is a temporary tree that are used to construct the data structure
        //since the entire tree need to be copied into device memory all pointer connections would fail
        root->build();

        //do a BFS to generate the actual tree
        cudaMallocHost(&tree.nodes, sizeof(AABBnode) * numNodes);

        queue<AABBNodeTemp*> currentLayer;
        queue<AABBNodeTemp*> nextLayer;
        currentLayer.push(root);

        while(count < numNodes){
            auto tmpCount = count;
            count += currentLayer.size();
            auto nxtCount = count;
            auto size = currentLayer.size();
            cout << "count: " << count << " layerSize" << size << endl;
            for(unsigned int ind = 0; ind < size; ind++){
                AABBNodeTemp* node = currentLayer.front();
                currentLayer.pop();

                tree.nodes[tmpCount + ind].minPoint = node->minPoint;
                tree.nodes[tmpCount + ind].maxPoint = node->maxPoint;
                tree.nodes[tmpCount + ind].isLeaf = node->isLeaf;
                tree.nodes[tmpCount + ind].trigIndex = node->trigIndex;

                if(node->left != nullptr){
                    tree.nodes[tmpCount + ind].left = nxtCount;
                    nextLayer.push(node->left);
                    nxtCount++;
                }

                if(node->right != nullptr){
                    tree.nodes[tmpCount + ind].right = nxtCount;
                    nextLayer.push(node->right);
                    nxtCount++;
                }

                cudaFreeHost(node);
            }

            currentLayer = nextLayer;
            nextLayer = queue<AABBNodeTemp*>();
        }

        cout<<"numNodes: "<<numNodes<<endl;
        ::exit(0);
    }

    TrigModel::TrigModel(const char *filename) {
        Model model;
        TinyGLTF loader;
        std::string err;
        std::string warn;

        if (string {filename}.find(".gltf") != string::npos) {
            loader.LoadASCIIFromFile(&model, &err, &warn, filename);
        } else if (string {filename}.find(".glb") != string::npos) {
            loader.LoadBinaryFromFile(&model, &err, &warn, filename);
        } else {
            std::cout << "File extension not supported" << std::endl;
            return;
        }

        if (!warn.empty()) {
            std::cout << "Warn: " << warn << std::endl;
        }

        if (!err.empty()) {
            std::cout << "Err: " << err << std::endl;
        }

        uint32_t numMeshes = model.meshes.size();
        uint32_t numPrimitives = 0;

        numVertices = 0;
        numTriangles = 0;

        for (auto& mesh : model.meshes) {
            for (auto& primitive : mesh.primitives) {
                const Accessor& vertexAccess = model.accessors[primitive.attributes["POSITION"]];
                numVertices += vertexAccess.count;

                const Accessor& indexAccess = model.accessors[primitive.indices];
                numTriangles += indexAccess.count / 3;

                numPrimitives++;
            }
        }

        //allocate memory for the meshes and primitives
        cudaMalloc(&meshesD, numMeshes * sizeof(MeshLabel));
        cudaMalloc(&primitivesD, numPrimitives * sizeof(PrimitiveLabel));
        cudaMallocHost(&meshes, numMeshes * sizeof(MeshLabel));
        cudaMallocHost(&primitives, numPrimitives * sizeof(PrimitiveLabel));
        assertCudaError();

        cudaMalloc(&verticesD, numVertices * sizeof(float3));
        cudaMallocHost(&vertices, numVertices * sizeof(float3));

        cudaMalloc(&normalsD, numVertices * sizeof(float3));
        cudaMallocHost(&normals, numVertices * sizeof(float3));

        cudaMalloc(&trianglesD, numTriangles * sizeof(triangle));
        cudaMallocHost(&triangles, numTriangles * sizeof(triangle));
        assertCudaError();

        uint32_t vertexProcIndex = 0;
        uint32_t triangleProcIndex = 0;
        uint32_t primitiveProcIndex = 0;
        uint32_t meshProcIndex = 0;

        //extract contents
        for (auto& mesh : model.meshes) {
            //create a mesh label
            meshes[meshProcIndex].MeshID = meshProcIndex;
            meshes[meshProcIndex].begPrimitiveIndex = primitiveProcIndex;
            meshes[meshProcIndex].endPrimitiveIndex = primitiveProcIndex + mesh.primitives.size();
            meshProcIndex++;

            for (auto& primitive : mesh.primitives) {

                //create a new primitive label
                primitives[primitiveProcIndex].PrimitiveID = primitiveProcIndex;
                primitives[primitiveProcIndex].begVertexIndex = vertexProcIndex;
                primitives[primitiveProcIndex].begTriangleIndex = triangleProcIndex;
                primitives[primitiveProcIndex].materialID = primitive.material;

                //load vertex data
                const Accessor& vertexAccess = model.accessors[primitive.attributes["POSITION"]];
                const BufferView& vertexBufferView = model.bufferViews[vertexAccess.bufferView];
                auto numV0 = vertexAccess.count;

                const Buffer& vertexBuffer = model.buffers[vertexBufferView.buffer];
                const auto* positions = reinterpret_cast<const float*>(&vertexBuffer.data[vertexBufferView.byteOffset + vertexAccess.byteOffset]);

                for(int i = 0; i < numV0; i++) {
                    //swap y and z for gltf orientation
                    float3 pos = make_float3(positions[i*3], positions[i*3+1], positions[i*3+2]);
                    vertices[vertexProcIndex + i] = pos;
                }

                //load normal data
                const Accessor& normalAccess = model.accessors[primitive.attributes["NORMAL"]];
                const BufferView& normalBufferView = model.bufferViews[normalAccess.bufferView];
                auto numN0 = normalAccess.count;

                const Buffer& normalBuffer = model.buffers[normalBufferView.buffer];
                const auto* normals0 = reinterpret_cast<const float*>(&normalBuffer.data[normalBufferView.byteOffset + normalAccess.byteOffset]);

                for(int i = 0; i < numN0; i++) {
                    float3 pos = make_float3(normals0[i*3], normals0[i*3+1], normals0[i*3+2]);
                    normals[vertexProcIndex + i] = pos;
                }

                primitives[primitiveProcIndex].endVertexIndex = vertexProcIndex + numV0;

                //load indices (triangle) data
                const Accessor& indexAccess = model.accessors[primitive.indices];
                const BufferView& indexBufferView = model.bufferViews[indexAccess.bufferView];
                auto numI0 = indexAccess.count;

                const Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

                if(indexAccess.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    const auto* indices = reinterpret_cast<const unsigned int*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccess.byteOffset]);
                    for(int i = 0; i < numI0; i+=3) {
                        auto tri = triangle(make_uint3(indices[i] + vertexProcIndex,
                                                       indices[i+1] + vertexProcIndex, indices[i+2] + vertexProcIndex));
                        triangles[triangleProcIndex + i/3] = tri;
                    }
                } else if(indexAccess.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    const auto* indices = reinterpret_cast<const unsigned short*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccess.byteOffset]);
                    for(int i = 0; i < numI0; i+=3) {
                        auto tri = triangle(make_uint3((uint)indices[i] + vertexProcIndex,
                                                       (uint)indices[i+1] + vertexProcIndex, (uint)indices[i+2] + vertexProcIndex));
                        triangles[triangleProcIndex + i/3] = tri;
                    }
                } else {
                    cerr << "Unsupported index component type" << endl;
                }

                vertexProcIndex += numV0;
                triangleProcIndex += numI0 / 3;
                primitives[primitiveProcIndex].endTriangleIndex = triangleProcIndex;

                primitiveProcIndex++;
            }
        }

        //apply transformations from nodes in the scene
        for(auto& node : model.nodes){
            if(!node.children.empty()){
                for(auto child : node.children){
                    if(model.nodes[child].mesh != -1) {
                        //apply transformations to the child
                        if (!node.scale.empty()) {
                            float3 scale = make_float3(node.scale[0], node.scale[1],
                                                       node.scale[2]);
                            for (auto i = meshes[model.nodes[child].mesh].begPrimitiveIndex;
                                 i < meshes[model.nodes[child].mesh].endPrimitiveIndex; i++) {
                                for (auto j = primitives[i].begVertexIndex; j < primitives[i].endVertexIndex; j++) {
                                    vertices[j] = hadamard3d(vertices[j], scale);
                                }
                            }
                        }

                        //translation
                        if (!node.translation.empty()) {
                            float3 translation = make_float3(node.translation[0],
                                                             node.translation[1],
                                                             node.translation[2]);
                            for (auto i = meshes[model.nodes[child].mesh].begPrimitiveIndex;
                                 i < meshes[model.nodes[child].mesh].endPrimitiveIndex; i++) {
                                for (auto j = primitives[i].begVertexIndex; j < primitives[i].endVertexIndex; j++) {
                                    vertices[j] = add3d(vertices[j], translation);
                                }
                            }
                        }
                    }
                }
            }

            if(node.mesh != -1){

                if(!node.scale.empty()){
                    float3 scale = make_float3(node.scale[0], node.scale[1], node.scale[2]);
                    for(auto i = meshes[node.mesh].begPrimitiveIndex; i < meshes[node.mesh].endPrimitiveIndex; i++){
                        for(auto j = primitives[i].begVertexIndex; j < primitives[i].endVertexIndex; j++){
                            vertices[j] = hadamard3d(vertices[j], scale);
                        }
                    }
                }

                //translation
                if(!node.translation.empty()){
                    float3 translation = make_float3(node.translation[0], node.translation[1], node.translation[2]);
                    for(auto i = meshes[node.mesh].begPrimitiveIndex; i < meshes[node.mesh].endPrimitiveIndex; i++){
                        for(auto j = primitives[i].begVertexIndex; j < primitives[i].endVertexIndex; j++){
                            vertices[j] = add3d(vertices[j], translation);
                        }
                    }
                }
            }
        }

        cudaMallocHost(&materials, model.materials.size() * sizeof(material));
        cudaMalloc(&materialsD, model.materials.size() * sizeof(material));
        int materialProcIndex = 0;
        for(auto material : model.materials){
            //load the material
            float3 baseColorFactor = {(float)material.pbrMetallicRoughness.baseColorFactor[0],
                                      (float)material.pbrMetallicRoughness.baseColorFactor[1],
                                      (float)material.pbrMetallicRoughness.baseColorFactor[2]};

            float metallicFactor = (float)material.pbrMetallicRoughness.metallicFactor;

            float roughnessFactor = (float)material.pbrMetallicRoughness.roughnessFactor;

            float3 emissiveFactor = {(float)material.emissiveFactor[0],
                                     (float)material.emissiveFactor[1],
                                     (float)material.emissiveFactor[2]};

            materials[materialProcIndex] = {emissiveFactor, baseColorFactor, metallicFactor, roughnessFactor};
            materialProcIndex++;
        }

        AABBTree tree = AABBTree();
        buildAABBTree(tree, triangles, numTriangles, vertices, numVertices);

        //copy to device
        cudaMemcpy(meshesD, meshes, numMeshes * sizeof(MeshLabel), cudaMemcpyHostToDevice);
        cudaMemcpy(primitivesD, primitives, numPrimitives * sizeof(PrimitiveLabel), cudaMemcpyHostToDevice);
        cudaMemcpy(verticesD, vertices, numVertices * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(normalsD, normals, numVertices * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(trianglesD, triangles, numTriangles * sizeof(triangle), cudaMemcpyHostToDevice);
        cudaMemcpy(materialsD, materials, model.materials.size() * sizeof(material), cudaMemcpyHostToDevice);
        assertCudaError();
    }
} // dylanrt