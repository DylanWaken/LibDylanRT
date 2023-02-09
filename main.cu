#include <iostream>
#include <cuda_gl_interop.h>

#include <cudaGL.h>

#include "Model/TrigModel.cuh"
#include "Renderer/VertexRenderer.cuh"
#include "Renderer/PhongShading.cuh"
#include "Display/PixelDisplay.cuh"
#include "Renderer/Transform.cuh"

using namespace std;
using namespace dylanrt;

int main(int argc, char** argv) {
    glutInit(&argc, argv);

    TrigModel model = TrigModel("/home/dylan/Documents/Resources/Models/car2/scene2.gltf");
    reorientModel(&model);
//    model.buildTree();

    int resX =2560;
    int resY =1600;

    EclipsePathO path = EclipsePathO(
            make_float3(-4, 4, 4),
            make_float3(0, 0, 0),
            make_float3(9.3, 0, 0),
            make_float3(0, 9.3, 0),
        2,
        1.200,
         1.920,
        resX,
        resY,
        0.01
        );

    float* imageD;
    cudaMalloc(&imageD, 3 * resX * resY * sizeof(float3));

    PixelDisplayGLUT display = PixelDisplayGLUT(resX, resY);
    //PixelDisplayGLFW display = PixelDisplayGLFW(resX, resY);
    createImage(imageD, pixelsD, resX, resY);

    CameraFrame* cam;
    for(int i = 0; i < 100000; i++) {
        cam = path.getNextCamD();

        cout << "Frame: " << i << endl;
//        renderEdges(model.verticesD, model.trianglesD, model.numTriangles, cam, imageD, resX*resY);

        transform(model.verticesD, model.screenVerticesD, model.screenSolvedD, cam, model.inrangeD, model.numVertices);

//        phongShading(model.materialsD, model.trianglesD, model.verticesD, model.tree.nodesD,nullptr, model.tree.numNodes
//        ,0,cam,imageD, resX*resY, make_float3(0,0,0));
        cout<<"Frame: "<<i<<endl;
        createImage(imageD, pixelsD, resX, resY);
        glutMainLoopEvent();
        cudaFree(cam);
    }

    return 0;
}
