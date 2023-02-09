//
// Created by dylan on 1/21/23.
//

#ifndef SIMPLEPATHTRACER_VERTEXRENDERER_CUH
#define SIMPLEPATHTRACER_VERTEXRENDERER_CUH

#include "CameraFrame.cuh"
#include "../Model/TrigModel.cuh"
#include "UtilFunctions.cuh"


namespace dylanrt {

    __host__ void assertCudaError();

    /**
     * we use the cramer's rule to solve the intersection of a ray and a plane
     * e + td = p1 + beta(p2 - p1) + gamma(p3 - p1)
     * solve the following linear system for t, gamma, beta
     *
     * [a_x - b_x  a_x - c_x  d_x] [beta ]   [a_x - e_x]
     * [a_y - b_y  a_y - c_y  d_y] [gamma] = [a_y - e_y]
     * [a_z - b_z  a_z - c_z  d_z] [t    ]   [a_z - e_z]
     *
     * @param rayOrigin e
     * @param rayDirection d
     * @param planePoint1 p1
     * @param planePoint2 p2
     * @param planePoint3 p3
     * @return [t, beta, gamma]
     */
    /*
     * [a_x - b_x  a_x - c_x  d_x] [beta ]   [a_x - e_x]
     * [a_y - b_y  a_y - c_y  d_y] [gamma] = [a_y - e_y]
     * [a_z - b_z  a_z - c_z  d_z] [t    ]   [a_z - e_z]
     */
    __device__ __forceinline__ float3 cramerIntersectSolver(float3 e, float3 d,
                                                            float3 a, float3 b,
                                                            float3 c){
        //determinant of left hand side matrix
        float M = (a.x - b.x) * ((a.y - c.y) * (d.z) - (d.y) * (a.z - c.z))
                  + (a.y - b.y) * ((d.x) * (a.z - c.z) - (a.x - c.x) * (d.z))
                  + (a.z - b.z) * ((a.x - c.x) * (d.y) - (d.x) * (a.y - c.y));

        float MInv = 1 / M;

        //first cramer determinant
        float beta = (a.x - e.x) * ((a.y - c.y) * (d.z) - (a.z - c.z) * (d.y))
                     + (a.y - e.y) * ((a.z - c.z) * (d.x) - (a.x - c.x) * (d.z))
                     + (a.z - e.z) * ((a.x - c.x) * (d.y) - (a.y - c.y) * (d.x));
        beta *= MInv;

        //second cramer determinant
        float gamma = (a.x - b.x) * ((a.y - e.y) * (d.z) - (a.z - e.z) * (d.y))
                      + (a.y - b.y) * ((a.z - e.z) * (d.x) - (a.x - e.x) * (d.z))
                      + (a.z - b.z) * ((a.x - e.x) * (d.y) - (a.y - e.y) * (d.x));
        gamma *= MInv;

        //third cramer determinant
        float t = (a.x - b.x) * ((a.y - c.y) * (a.z - e.z) - (a.z - c.z) * (a.y - e.y))
                  + (a.y - b.y) * ((a.z - c.z) * (a.x - e.x) - (a.x - c.x) * (a.z - e.z))
                  + (a.z - b.z) * ((a.x - c.x) * (a.y - e.y) - (a.y - c.y) * (a.x - e.x));
        t *= MInv;

        return make_float3(beta, gamma, t);
    }

    /**
     * whether the point of intersect is in range of the rectangle (or parallelogram)
     * can be solved by checking if beta and gamma are in range [0, 1]
     * (these parameters are returned from the cramer solver)
     *
     * @param beta beta (b-a)
     * @param gamma gamma (c-a)
     * @return
     */
    __device__ __forceinline__ bool rectInRange(float beta,float gamma){
        return beta >= 0 && beta <= 1 && gamma >= 0 && gamma <= 1;
    }

    __device__ __forceinline__ bool trigInRange(float beta, float gamma){
        return beta >= 0 && gamma >= 0 && beta + gamma <= 1;
    }

    void renderVertices(float3 *vertices, unsigned int numVertices, CameraFrame* cameraFrame,
                        float* imagePlane,  int pixelCount);

    struct solvedTrig{
        float3 scrPoint1;
        float3 scrPoint2;
        float3 scrPoint3;
    };

    void renderEdges(float3 *vertices, triangle* trigs, unsigned int numTrigs,
                     CameraFrame* cameraFrame, float* imagePlane,  int pixelCount);

} // dylanrt

#endif //SIMPLEPATHTRACER_VERTEXRENDERER_CUH
