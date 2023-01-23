//
// Created by dylan on 1/21/23.
//

#ifndef SIMPLEPATHTRACER_CAMERAFRAME_CUH
#define SIMPLEPATHTRACER_CAMERAFRAME_CUH

#include "../Model/TrigModel.cuh"

namespace dylanrt {

    struct Ray{
        float3 origin;
        float3 direction;

        __device__ __host__ Ray(float3 origin, float3 direction) : origin(origin), direction(direction) {}
    };

    //this is the camera frame
    /*
     * direction vector (unit) (-w) that gives the direction of observation
     * - NOTE: direction vector is opposite of the view direction
     * basis vector u, v (unit) are orthogonal to w and form the basis of the camera frame (orientation)
     * - U is horizontal, V is vertical
     * position vector (e) is the position of the camera (relative to model/scene origin)
     * d is the distance from the camera to the image plane
     * tb is the height of the image plane
     * lr is the width of the image plane
     */
    class CameraFrame {
    public:
        float3 directionW;
        float3 basisU;
        float3 basisV;
        float3 positionE;

        float distanceD;
        float height;
        float width;

        float3 imgTopLeft{};
        float3 imgTopRight{};
        float3 imgBottomLeft{};
        float3 imgBottomRight{};

        int resolutionX;
        int resolutionY;

        //solve the four image plane corners given the camera frame
        __host__ void solveImagePlane();

        //create the camera frame
        __host__ CameraFrame(float3 directionW, float3 basisU, float3 basisV,
                                        float3 positionE, float distanceD, float topBottomTB,
                                        float leftRightLR, int resolutionX, int resolutionY);

        //create the camera frame with rotation angle for U V (in degrees)
        __host__ CameraFrame(float3 directionW0, float3 positionE, float distanceD,
                                        float topBottomTB, float leftRightLR, int resolutionX,
                                        int resolutionY, float rotationAngleUV);
    };

    struct CameraPath{
    public:
        virtual CameraFrame getNextCam() = 0;
        virtual CameraFrame* getNextCamD() = 0;
    };

    struct EclipsePathO : public CameraPath{

        float distanceD;
        float topBottomTB;
        float leftRightLR;
        int resolutionX;
        int resolutionY;

        float3 center;
        float3 viewCenter;

        float3 u;  // long axis
        float3 v;  // short axis
        float step;

        float t = 0;

        CameraFrame getNextCam() override;
        CameraFrame* getNextCamD() override;

        EclipsePathO(float3 center, float3 viewCenter, float3 u, float3 v, float distanceD,
                     float topBottomTB, float leftRightLR, int resolutionX, int resolutionY, float step) :
                     center(center), viewCenter(viewCenter), u(u), v(v), distanceD(distanceD),
                     topBottomTB(topBottomTB), leftRightLR(leftRightLR), resolutionX(resolutionX),
                     resolutionY(resolutionY), step(step){};
    };
}


#endif //SIMPLEPATHTRACER_CAMERAFRAME_CUH
