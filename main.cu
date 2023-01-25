#include <iostream>
#include <cuda_gl_interop.h>

#include <cudaGL.h>

#include "Model/TrigModel.cuh"
#include "Renderer/VertexRenderer.cuh"
#include "Display/PixelDisplay.cuh"

using namespace std;
using namespace dylanrt;

int main(int argc, char** argv) {
    glutInit(&argc, argv);

    TrigModel model = TrigModel("/media/dylan/DylanFiles/Resources/Models/car2/scene2.gltf");
    reorientModel(&model);

    int resX = 1440;
    int resY = 960;

    EclipsePathO path = EclipsePathO(
        make_float3(-4, 4, 4),
        make_float3(0, 0, 0),
        make_float3(9.3, 0, 0),
        make_float3(0, 9.3, 0),
        0.7,
        1*0.480,
         1*0.720,
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
        renderVertices(model.verticesD, model.numVertices, cam, imageD, resX * resY);
        createImage(imageD, pixelsD, resX, resY);
        glutPostRedisplay();
        //this_thread::sleep_for(chrono::milliseconds(1000/60));
        cudaFree(cam);
    }

    tr.join();
    return 0;
}
