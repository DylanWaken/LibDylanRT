//
// Created by dylan on 1/21/23.
//

#include <cassert>
#include "CameraFrame.cuh"

namespace dylanrt{

     __host__ void CameraFrame::solveImagePlane() {

        //get the image plane center (perpendicular to the view direction)
        float3 planeCenter = add3d(positionE, scale3d(directionW, (-distanceD)));

         //get the up left corner of the image plane
        imgTopLeft = subtract3d(planeCenter, scale3d(basisU, width / 2));
        imgTopLeft = add3d(imgTopLeft, scale3d(basisV, height / 2));

        //get the up right corner of the image plane
        imgTopRight = add3d(planeCenter, scale3d(basisU, width / 2));
        imgTopRight = add3d(imgTopRight, scale3d(basisV, height / 2));

        //get the bottom left corner of the image plane
        imgBottomLeft = subtract3d(planeCenter, scale3d(basisU, width / 2));
        imgBottomLeft = subtract3d(imgBottomLeft, scale3d(basisV, height / 2));

        //get the bottom right corner of the image plane
        imgBottomRight = add3d(planeCenter, scale3d(basisU, width / 2));
        imgBottomRight = subtract3d(imgBottomRight, scale3d(basisV, height / 2));

    }

     __host__ CameraFrame::CameraFrame(float3 directionW, float3 basisU, float3 basisV, float3 positionE,
                                                 float distanceD, float topBottomTB, float leftRightLR, int resolutionX,
                                                 int resolutionY)

         : directionW(normalize3d(directionW)),
           basisU(normalize3d(basisU)),
           basisV(normalize3d(basisV)),
           positionE(positionE),
           distanceD(distanceD),
           height(topBottomTB),
           width(leftRightLR),
           resolutionX(resolutionX),
           resolutionY(resolutionY){

        solveImagePlane();
    }

    __host__ CameraFrame::CameraFrame(float3 directionW0, float3 positionE, float distanceD,
                                                 float topBottomTB, float leftRightLR, int resolutionX, int resolutionY,
                                                 float rotationAngleUV)

            : directionW(normalize3d(directionW0)),
              positionE(positionE),
              distanceD(distanceD),
              height(topBottomTB),
              width(leftRightLR),
              resolutionX(resolutionX),
              resolutionY(resolutionY){


        //first solve the default u vector with zero z component (cross w and z)
        float3 defaultU = cross(directionW, make_float3(0, 0, -1));

        //rotate the default u vector by the rotation angle on the z direction
        basisU = normalize3d(rotate3dz(defaultU, rotationAngleUV));

        //solve the v vector by cross w and u
        basisV = normalize3d(cross(basisU, directionW));

        solveImagePlane();
    }

    CameraFrame EclipsePathO::getNextCam() {
        //the parametric equation of the eclipse is
        //𝐱(𝑡)=𝐜+(cos𝑡)𝐮+(sin𝑡)𝐯
        float3 pos = add3d(add3d(center, scale3d(u, cosf(t))), scale3d(v, sinf(t)));
        float3 viewDir = subtract3d(pos, viewCenter);

        t += step;
        t = fmodf(t, 2 * M_PI);

        //the new camera frame
        return CameraFrame(viewDir, pos, distanceD, topBottomTB, leftRightLR, resolutionX, resolutionY, 0);
    }

    CameraFrame* EclipsePathO::getNextCamD() {
        auto cam = getNextCam();
        CameraFrame* camD;
        cudaMalloc(&camD, sizeof(CameraFrame));
        cudaMemcpy(camD, &cam, sizeof(CameraFrame), cudaMemcpyHostToDevice);
        return camD;
    }
} // dylanrt