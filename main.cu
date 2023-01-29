#include <iostream>
#include <cuda_gl_interop.h>

#include <cudaGL.h>

#include "Model/TrigModel.cuh"
#include "Renderer/VertexRenderer.cuh"
#include "Renderer/PhongShading.cuh"
#include "Display/PixelDisplay.cuh"

using namespace std;
using namespace dylanrt;



int main(int argc, char** argv) {
    glutInit(&argc, argv);

    TrigModel model = TrigModel("/media/dylan/DylanFiles/Resources/Models/Sphere/Sphere.glb");
    reorientModel(&model);
    model.buildTree();

    int resX = 1920;
    int resY = 1200;

    EclipsePathO path = EclipsePathO(
        make_float3(0, 0, 4),
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

    PixelDisplay display = PixelDisplay(resX, resY);
    createImage(imageD, pixelsD, resX, resY);

    auto tr = display.start();

    CameraFrame* cam;
    for(int i = 0; i < 100000; i++) {
        cam = path.getNextCamD();

        //cout << "Frame: " << i << endl;
        phongShading(model.materialsD, model.trianglesD, model.verticesD, model.tree.nodesD,nullptr, model.tree.numNodes
        ,0,cam,imageD, resX*resY, make_float3(0,0,0));
        createImage(imageD, pixelsD, resX, resY);
        glutPostRedisplay();
        cudaFree(cam);
    }

    tr.join();
    return 0;
}
