//
// Created by dylan on 1/20/23.
//

#include "TrigModel.cuh"
#include "../Renderer/UtilFunctions.cuh"
#include "TinyGLTF/tiny_gltf.h"

using namespace tinygltf;
namespace dylanrt {

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

        size_t numMeshes = model.meshes.size();
        size_t numPrimitives = 0;

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

        size_t vertexProcIndex = 0;
        size_t triangleProcIndex = 0;
        size_t primitiveProcIndex = 0;
        size_t meshProcIndex = 0;

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

                //get primitive bounding box
                primitives[primitiveProcIndex].max = make_float3(vertexAccess.maxValues[0], vertexAccess.maxValues[1], vertexAccess.maxValues[2]);
                primitives[primitiveProcIndex].min = make_float3(vertexAccess.minValues[0], vertexAccess.minValues[1], vertexAccess.minValues[2]);

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


                vertexProcIndex += numV0;
                primitives[primitiveProcIndex].endVertexIndex = vertexProcIndex;

                //load indices (triangle) data
                const Accessor& indexAccess = model.accessors[primitive.indices];
                const BufferView& indexBufferView = model.bufferViews[indexAccess.bufferView];
                auto numI0 = indexAccess.count;

                const Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
                const auto* indices = reinterpret_cast<const unsigned short*>
                        (&indexBuffer.data[indexBufferView.byteOffset + indexAccess.byteOffset]);

                for(int i = 0; i < numI0; i+=3) {
                    auto tri = triangle(make_ushort3(indices[i], indices[i+1], indices[i+2]));
                    triangles[triangleProcIndex + i/3] = tri;
                }
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

        //copy to device
        cudaMemcpy(meshesD, meshes, numMeshes * sizeof(MeshLabel), cudaMemcpyHostToDevice);
        cudaMemcpy(primitivesD, primitives, numPrimitives * sizeof(PrimitiveLabel), cudaMemcpyHostToDevice);
        cudaMemcpy(verticesD, vertices, numVertices * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(normalsD, normals, numVertices * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(trianglesD, triangles, numTriangles * sizeof(triangle), cudaMemcpyHostToDevice);
        assertCudaError();
    }
} // dylanrt