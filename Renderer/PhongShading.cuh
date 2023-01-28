//
// Created by dylan on 1/25/23.
//

#ifndef SIMPLEPATHTRACER_PHONGSHADING_CUH
#define SIMPLEPATHTRACER_PHONGSHADING_CUH

#include "UtilFunctions.cuh"
#include "VertexRenderer.cuh"

namespace dylanrt {

    /**
     * @brief Phong Shading
     *
     * The earliest shading models, included diffuse and specular shading.
     * The math formula is given below in latex formats
     *
     * I = I_{a} + k_{d} \sum_{j} (\hat{N_{p}} \cdot \hat{L}_{j}) I_{d,j} + k_{s}
     * \sum_{j}( \hat{V_{p}}\cdot \hat{R_{j}})^n I_{s,j}
     *
     * - $I$ is the reflected intensity of a point toward viewer
     * - $I_{a}$ is the reflection from ambient light
     * - $I_{d}$ is the strength of incoming light
     * - $I_{s}$ is the strength of reflected light

     * - $k_{d}$ the diffuse reflection constant
     * - $k_{s}$ is the specualr reflection constant, which replace the original function $W$

     * - $N_p$ is the approximate normal vector.
     * - $R$ is the perfectly reflected light vector at $P$
     * - $V_{p}$ is the direction pointing towards the viewer
     * - $L$ is the direction toward the light source
     *
     * - j iterates through all the light sources
     *
     * @param primitives
     * @param materials Note: since gltf stores material in the manner of metalness and roughness,
     *                 we would use it as the specular and diffused constants respectively.
     * @param trigs
     * @param vertices
     * @param lights
     * @param numTrigs
     * @param numLights
     * @param cameraFrame
     * @param imagePlane
     * @param numPixls
     * @param Ia
     */
    void phongShading(PrimitiveLabel* primitives, material* materials, triangle* trigs, float3* vertices, pointLight* lights,
                      unsigned int numTrigs, unsigned int numLights, CameraFrame* cameraFrame,float* imagePlane,
                      unsigned int numPixls, float3 ambientLight);

} // dylanrt

#endif //SIMPLEPATHTRACER_PHONGSHADING_CUH
