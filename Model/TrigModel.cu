//
// Created by dylan on 1/20/23.
//

#include "TrigModel.cuh"
#include "TinyGLTF/tiny_gltf.h"

using namespace tinygltf;
namespace dylanrt {

    TrigModel::TrigModel(const char *filename) {
        Model model;
        TinyGLTF loader;
        std::string err;
        std::string warn;

        bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);

        if (!warn.empty()) {
            std::cout << "Warn: " << warn << std::endl;
        }

        if (!err.empty()) {
            std::cout << "Err: " << err << std::endl;
        }

        //load the vertices
        const Accessor& accessor = model.accessors[model.meshes[0].primitives[0].attributes["POSITION"]];
        const BufferView& bufferView = model.bufferViews[accessor.bufferView];

        const Buffer& buffer = model.buffers[bufferView.buffer];
        const auto* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

        numVertices = accessor.count;
        cudaMallocHost(&vertices, numVertices * sizeof(float3));

        for(int i = 0; i < numVertices; i++) {
            float3 pos = make_float3(positions[i*3], positions[i*3+1], positions[i*3+2]);
            vertices[i] = pos;
        }

        cudaMalloc(&verticesD, numVertices * sizeof(float3));
        cudaMemcpy(verticesD, vertices, numVertices * sizeof(float3), cudaMemcpyHostToDevice);

        //load the vertex normals
        const Accessor& accessor1 = model.accessors[model.meshes[0].primitives[0].attributes["NORMAL"]];
        const BufferView& bufferView1 = model.bufferViews[accessor1.bufferView];

        const Buffer& buffer1 = model.buffers[bufferView1.buffer];
        auto* normals0 = reinterpret_cast<const float*>(&buffer1.data[bufferView1.byteOffset + accessor1.byteOffset]);

        cudaMallocHost(&normals, numVertices * sizeof(float3));

        for(int i = 0; i < numVertices; i++) {
            float3 norm = make_float3(normals0[i*3], normals0[i*3+1], normals0[i*3+2]);
            normals[i] = norm;
        }

        cudaMalloc(&normalsD, numVertices * sizeof(float3));
        cudaMemcpy(normalsD, normals, numVertices * sizeof(float3), cudaMemcpyHostToDevice);


        //load the triangles
        const Accessor& accessor2 = model.accessors[model.meshes[0].primitives[0].indices];
        const BufferView& bufferView2 = model.bufferViews[accessor2.bufferView];

        const Buffer& buffer2 = model.buffers[bufferView2.buffer];
        auto* indices0 = reinterpret_cast<const unsigned short*>(&buffer2.data[bufferView2.byteOffset + accessor2.byteOffset]);

        numTriangles = accessor2.count / 3;
        cudaMallocHost(&triangles, numTriangles * sizeof(triangle));

        for(int i = 0; i < numTriangles; i++) {
            triangles[i] = triangle(make_ushort3(indices0[i*3], indices0[i*3+1], indices0[i*3+2]));
        }

        cudaMalloc(&trianglesD, numTriangles * sizeof(triangle));
        cudaMemcpy(trianglesD, triangles, numTriangles * sizeof(triangle), cudaMemcpyHostToDevice);
    }
} // dylanrt