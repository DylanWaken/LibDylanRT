//
// Created by dylan on 1/25/23.
//

#include "PhongShading.cuh"
#include "UtilFunctions.cuh"

namespace dylanrt {

    //we use slab method to check if a ray intersects with a box
    //basically we treat the box as pairs of parellel planes and gradually clip the ray
    //if the ray is still alive after clipping, then it intersects with the box
    __device__ __forceinline__ bool rayBoxIntersect(float3 max, float3 min, float3 e, float3 d){
        float t1x = (min.x - e.x) / d.x;
        float t2x = (max.x - e.x) / d.x;

        float t1y = (min.y - e.y) / d.y;
        float t2y = (max.y - e.y) / d.y;

        float t1z = (min.z - e.z) / d.z;
        float t2z = (max.z - e.z) / d.z;

        float tmin = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
        float tmax = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));

        return tmax >= tmin;
    }

    template<const int BM>
    __global__ void phongShadingD(PrimitiveLabel* primitives, material* materials, triangle* trigs, float3* vertices, pointLight* lights,
                                  unsigned int numPrimitives, unsigned int numTrigs, unsigned int numLights, CameraFrame* cameraFrame,float* imagePlane,
                                  unsigned int numPixls, float3 ambientLight){


    }

    void phongShading(PrimitiveLabel* primitives, material* materials, triangle* trigs, float3* vertices, pointLight* lights,
                      unsigned int numTrigs, unsigned int numLights, CameraFrame* cameraFrame,float* imagePlane,
                      unsigned int numPixls, float3 ambientLight){

    }
} // dylanrt