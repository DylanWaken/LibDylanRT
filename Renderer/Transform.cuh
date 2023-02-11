//
// Created by dylan on 2/7/23.
//

#ifndef SIMPLEPATHTRACER_TRANSFORM_CUH
#define SIMPLEPATHTRACER_TRANSFORM_CUH

#include "../Model/TrigModel.cuh"
#include "VertexRenderer.cuh"

namespace dylanrt {
/**
 * We transform 3d model space vertices to 2d screen space vertices.
 */

    void transform(float3* vertices, float3* screenVertices, float3* screenSolved,CameraFrame* cam, bool* inrange, unsigned int numVertices);
}

#endif //SIMPLEPATHTRACER_TRANSFORM_CUH
