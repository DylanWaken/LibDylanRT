cmake_minimum_required(VERSION 3.24)
add_compile_options(-g -G)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(SimplePathTracer CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(--keep -Xptxas -v -ftz=true -prec-div=false -prec-sqrt=false)

add_executable(SimplePathTracer
        main.cu

        Model/TinyGLTF/tiny_gltf.h
        Model/TinyGLTF/stb_image.h
        Model/TinyGLTF/stb_image_write.h
        Model/TinyGLTF/json.hpp

        Model/TrigModel.cu
        Model/TrigModel.cuh

        Renderer/VertexRenderer.cu
        Renderer/VertexRenderer.cuh
        Renderer/CameraFrame.cu
        Renderer/CameraFrame.cuh

        Display/PixelDisplay.cu
        Display/PixelDisplay.cuh
        Renderer/UtilFunctions.cu
        Renderer/UtilFunctions.cuh
        Renderer/PhongShading.cu Renderer/PhongShading.cuh Renderer/Transform.cu Renderer/Transform.cuh)

target_link_libraries(SimplePathTracer glfw3 glut X11 Xrandr Xi Xxf86vm GL GLU GLEW cudnn)

set_target_properties(SimplePathTracer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CMAKE_CUDA_ARCHITECTURES 89)
