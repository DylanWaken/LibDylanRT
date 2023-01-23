//
// Created by dylan on 1/22/23.
//

#include <GL/gl.h>
#include <cudaGL.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include "../Renderer/UtilFunctions.cuh"
#include <thread>

#ifndef SIMPLEPATHTRACER_PIXELDISPLAY_CUH
#define SIMPLEPATHTRACER_PIXELDISPLAY_CUH

namespace dylanrt {

    extern int windowH;
    extern int windowW;
    extern unsigned char* pixels;
    extern unsigned char* pixelsD;

    struct PixelDisplay {

        PixelDisplay(int w, int h) {
            cudaMallocHost(&pixels, w * h * sizeof(unsigned char) * 3);
            cudaMalloc(&pixelsD, w * h * sizeof(unsigned char) * 3);
            assertCudaError();

            windowH = h;
            windowW = w;

            glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
            glutInitWindowSize(w, h);
            glutCreateWindow("Render Preview");

            glutDisplayFunc([]() {
                glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
                cudaMemcpy(pixels, pixelsD, windowW * windowH * sizeof(unsigned char) * 3, cudaMemcpyDeviceToHost);
                glDrawPixels(windowW, windowH, GL_RGB, GL_UNSIGNED_BYTE, pixels);
                glFlush();
                cudaMemset(pixels, 0, windowW * windowH * sizeof(unsigned char) * 3);
                glutPostRedisplay();
            });
        }

        //create mainloop thread
        thread start() {
            return thread(glutMainLoop);
        }
    };


} // dylanrt

#endif //SIMPLEPATHTRACER_PIXELDISPLAY_CUH
