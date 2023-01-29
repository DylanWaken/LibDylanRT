//
// Created by dylan on 1/20/23.
//

#ifndef SIMPLEPATHTRACER_TRIGMODEL_CUH
#define SIMPLEPATHTRACER_TRIGMODEL_CUH

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define subtract3d(a,b) (make_float3(a.x - b.x, a.y - b.y, a.z - b.z))
#define add3d(a,b) (make_float3(a.x + b.x, a.y + b.y, a.z + b.z))
#define scale3d(a, c0) (make_float3(a.x * c0, a.y * c0, a.z * c0))
#define norm3d(a) (sqrtf(a.x*a.x + a.y*a.y + a.z*a.z))
#define negate(a) (make_float3(-a.x, -a.y, -a.z))

#define dot3d(a,b) (a.x*b.x + a.y*b.y + a.z*b.z)
#define cross(a,b) (make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x))
#define normalize3d(a) (scale3d(a, 1/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z)))
#define hadamard3d(a,b) (make_float3(a.x*b.x, a.y*b.y, a.z*b.z))

#define rotate3dx(a, angle) (make_float3( \
     1 * a.x + 0 * a.y + 0 * a.z \
    ,0 * a.x + cosf(angle) * a.y - sinf(angle) * a.z \
    ,0 * a.x + sinf(angle) * a.y + cosf(angle) * a.z \
))

#define rotate3dy(a, angle) (make_float3( \
    cosf(angle) * a.x + 0 * a.y + sinf(angle) * a.z \
    ,0 * a.x + 1 * a.y + 0 * a.z \
    ,-sinf(angle) * a.x + 0 * a.y + cosf(angle) * a.z \
))

#define rotate3dz(a, angle) (make_float3( \
    cosf(angle) * a.x - sinf(angle) * a.y + 0 * a.z \
    ,sinf(angle) * a.x + cosf(angle) * a.y + 0 * a.z \
    ,0 * a.x + 0 * a.y + 1 * a.z \
))

#define EPSILON 0.0001f

using namespace std;
namespace dylanrt {

    //standard triangles
    struct triangle {
        uint3 indices{};
        float3 normal{};
        unsigned int materialID;

        //a triangle defined by 3 vertices
        __device__ __host__ triangle(uint3 indices)
                :indices(indices) {
        }
    };

    //2d matrices
    struct matrix2d {
        float* data;
        int h;
        int w;

        __device__ __host__ matrix2d(float* data, int h, int w) : data(data), h(h), w(w) {}

        __device__ __host__ float& operator()(int i, int j) const {
            return data[i*w + j];
        }

        __device__ __host__ float* operator[](int i) const {
            return data + i*w;
        }
    };

    struct PrimitiveLabel{
        unsigned int PrimitiveID;
        float3 maxPoint;
        float3 minPoint;
        uint32_t begVertexIndex;
        uint32_t endVertexIndex;
        uint32_t begTriangleIndex;
        uint32_t endTriangleIndex;

        unsigned int materialID;
    };

    struct MeshLabel{
        unsigned int MeshID;
        unsigned int begPrimitiveIndex;
        unsigned int endPrimitiveIndex;
    };

    struct material{
        float3 emissiveFactor;
        float3 baseColorFactor;
        float metallicFactor;
        float roughnessFactor;
    };

    struct AABBnode{
        float3 maxPoint;
        float3 minPoint;

        bool isLeaf;

        //if leaf the node would store the triangle indices
        uint32_t trigIndex;

        //if not leaf the node would store the child indices
        //the childs are also AABBnodes
        uint32_t left;
        uint32_t right;
    };

    struct AABBTree{
        AABBnode* nodes;
        AABBnode* nodesD;
        uint32_t numNodes;
    };

    void buildAABBTree(AABBTree& tree, triangle* trigs, uint32_t numTrigs,
                       float3* vertices, uint32_t numVertices);

    //store a minimal model (with no subcomponents, textures, etc. only vertices and triangles)
    struct TrigModel {
        //The vertices in Host memory
        float3* vertices;
        float3* normals;
        triangle* triangles;
        material* materials;

        PrimitiveLabel* primitives;
        MeshLabel* meshes;

        PrimitiveLabel* primitivesD;
        MeshLabel* meshesD;

        //The vertices in Device memory
        float3* verticesD;
        float3* normalsD;
        triangle* trianglesD;
        material* materialsD;

        uint32_t numVertices;
        uint32_t numTriangles;

        AABBTree tree;

        //create a minimal model from a file:
        explicit TrigModel(const char* filename);

        void buildTree();
    };

    struct pointLight{
        float3 position;
        float3 baseColor;
        float intensity;
    };

} // dylanrt

#endif //SIMPLEPATHTRACER_TRIGMODEL_CUH
