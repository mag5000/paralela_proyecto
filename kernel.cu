#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <chrono>

using namespace std;


// ==========================================================================================================================================================================================
// ======================================                                           ALGORITMOS CUDA V1                                   ===================================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================

// Función para convertir una imagen en escala de grises 
__global__ void convertToGrayscaleCUDA(const unsigned char* image, int width, int height, int channels, unsigned char* grayImage) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height) {
        for (int col = 0; col < width; ++col) {

            int grayOffset = row * width + col;
            int offset = (row * width + col) * channels;

            // Convertir píxel a escala de grises (promedio de los componentes RGB)
            if (channels == 3) {
                grayImage[grayOffset] = (unsigned char)((image[offset] + image[offset + 1] + image[offset + 2]) / 3);
            }
            else if (channels == 1) {
                grayImage[grayOffset] = image[offset]; // Si ya está en escala de grises, copiar el valor directamente
            }

        }
    }
    //__syncthreads();
}

// Función para calcular el LBP para un punto en una imagen en escala de grises con un parche de 5x5
__device__ int calculateLBPCUDA(const unsigned char* image, int width, int x, int y) {
    int center = image[y * width + x];
    int lbpCode = 0;
    const int numNeighbors = 24;

    // Vecinos en sentido horario 
    int neighborOffsets[24][2] = {
        {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
        {2, -1}, {2, 0}, {2, 1}, {2, 2},
        {1, 2}, {0, 2}, {-1, 2}, {-2, 2},
        {-2, 1}, {-2, 0}, {-2, -1},
        {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}
    };

    for (int i = 0; i < numNeighbors; ++i) {
        int xOffset = neighborOffsets[i][0];
        int yOffset = neighborOffsets[i][1];
        int xx = x + xOffset;
        int yy = y + yOffset;

        // Comprueba los límites de la imagen
        if (xx >= 0 && xx < width && yy >= 0) {
            int neighborValue = image[yy * width + xx];
            // Calcula si el vecino es mayor o igual al centro
            bool isGreaterOrEqual = (neighborValue >= center);
            // Desplaza el bit 1 a la posición correspondiente
            int shiftedBit = isGreaterOrEqual << i;
            // Combina el bit en el código LBP
            lbpCode |= shiftedBit;
        }
    }

    return lbpCode;
}

// Kernel CUDA para verificar si un píxel es un punto clave FAST, calcular su descriptor LBP y escribirlo en una matriz
__global__ void findKeyPointsAndDescriptorsCUDA(const unsigned char* image, int width, int height, int channels, int threshold, int* keypoints, int* descriptors) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height - 3 && row >= 3) {
        for (int x = 3; x < width - 3; ++x) {
            // Obtener el valor del píxel central
            unsigned char center = image[(row * width + x) * channels];

            // Definir los desplazamientos de los píxeles vecinos
            int pixelOffsets[16][2] = {
                {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}, {0, 3}, {1, 3}, {2, 2}, {3, 1},
                {3, 0}, {3, -1}, {2, -2}, {1, -3}, {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}
            };

            // Contadores para píxeles más brillantes y más oscuros
            int brighterCount = 0;
            int darkerCount = 0;

            // Calcular el número de píxeles más brillantes y más oscuros en la vecindad del píxel central
            for (int i = 0; i < 16; ++i) {
                int xOffset = pixelOffsets[i][0];
                int yOffset = pixelOffsets[i][1];
                int xx = x + xOffset;
                int yy = row + yOffset;
                if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                    unsigned char neighbor = image[(yy * width + xx) * channels];
                    if (neighbor > center + threshold) {
                        brighterCount++;
                    }
                    else if (neighbor < center - threshold) {
                        darkerCount++;
                    }
                }
            }

            // Si la mayoría de los píxeles vecinos son más brillantes o más oscuros, se considera un punto clave
            if (brighterCount >= 9 || darkerCount >= 9) {
                // Incrementar el contador de puntos clave y guardar las coordenadas del punto clave
                int idx = atomicAdd(&keypoints[0], 1);
                keypoints[idx * 2 + 1] = x;
                keypoints[idx * 2 + 2] = row;

                // Calcular el descriptor LBP para el punto clave y guardarlo en la matriz de descriptores
                int descriptor = calculateLBPCUDA(image, width, x, row);
                descriptors[idx] = descriptor;

            }

        }
    }

    //__syncthreads();
}




// ==========================================================================================================================================================================================
// ======================================                                          FIN CUDA V1                                     ==========================================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================

// ==========================================================================================================================================================================================
// ======================================                                     ALGORITMOS CUDA V2                                          ===================================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================



// Función para convertir una imagen en escala de grises 
__global__ void convertToGrayscaleCUDAV2(const unsigned char* image, int width, int height, int channels, unsigned char* grayImage) {

    // Calcular las coordenadas globales dentro de la imagen para este hilo
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (y < height) {
        if (x < width) {

            int grayOffset = y * width + x;
            int offset = (y * width + x) * channels;

            // Convertir píxel a escala de grises (promedio de los componentes RGB)
            if (channels == 3) {
                grayImage[grayOffset] = (unsigned char)((image[offset] + image[offset + 1] + image[offset + 2]) / 3);
            }
            else if (channels == 1) {
                grayImage[grayOffset] = image[offset]; // Si ya está en escala de grises, copiar el valor directamente
            }

        }
    }
    //__syncthreads();
}

// Kernel CUDA para verificar si un píxel es un punto clave FAST, calcular su descriptor LBP y escribirlo en una matriz
__global__ void findKeyPointsAndDescriptorsCUDAV2(const unsigned char* image, int width, int height, int channels, int threshold, int* keypoints, int* descriptors) {


    // Calcular las coordenadas globales dentro de la imagen para este hilo
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (y < height - 3 && y >= 3) {
        if (x < width - 3 && x >= 3) {
            // Obtener el valor del píxel central
            unsigned char center = image[(y * width + x) * channels];

            // Definir los desplazamientos de los píxeles vecinos
            int pixelOffsets[16][2] = {
                {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}, {0, 3}, {1, 3}, {2, 2}, {3, 1},
                {3, 0}, {3, -1}, {2, -2}, {1, -3}, {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}
            };

            // Contadores para píxeles más brillantes y más oscuros
            int brighterCount = 0;
            int darkerCount = 0;

            // Calcular el número de píxeles más brillantes y más oscuros en la vecindad del píxel central
            for (int i = 0; i < 16; ++i) {
                int xOffset = pixelOffsets[i][0];
                int yOffset = pixelOffsets[i][1];
                int xx = x + xOffset;
                int yy = y + yOffset;
                if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                    unsigned char neighbor = image[(yy * width + xx) * channels];
                    if (neighbor > center + threshold) {
                        brighterCount++;
                    }
                    else if (neighbor < center - threshold) {
                        darkerCount++;
                    }
                }
            }

            // Si la mayoría de los píxeles vecinos son más brillantes o más oscuros, se considera un punto clave
            if (brighterCount >= 9 || darkerCount >= 9) {
                // Incrementar el contador de puntos clave y guardar las coordenadas del punto clave
                int idx = atomicAdd(&keypoints[0], 1);
                keypoints[idx * 2 + 1] = x;
                keypoints[idx * 2 + 2] = y;

                // Calcular el descriptor LBP para el punto clave y guardarlo en la matriz de descriptores
                int descriptor = calculateLBPCUDA(image, width, x, y);
                descriptors[idx] = descriptor;

            }

        }
    }
    //__syncthreads();
}

// Función para calcular el LBP para un punto en una imagen en escala de grises con un parche de 5x5
__device__ int calculateLBPCUDAV2(const unsigned char* image, int width, int x, int y) {
    int center = image[y * width + x];
    int lbpCode = 0;
    const int numNeighbors = 24;

    // Vecinos en sentido horario 
    int neighborOffsets[24][2] = {
        {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
        {2, -1}, {2, 0}, {2, 1}, {2, 2},
        {1, 2}, {0, 2}, {-1, 2}, {-2, 2},
        {-2, 1}, {-2, 0}, {-2, -1},
        {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}
    };

    for (int i = 0; i < numNeighbors; ++i) {
        int xOffset = neighborOffsets[i][0];
        int yOffset = neighborOffsets[i][1];
        int xx = x + xOffset;
        int yy = y + yOffset;

        // Comprueba los límites de la imagen
        if (xx >= 0 && xx < width && yy >= 0) {
            int neighborValue = image[yy * width + xx];
            // Calcula si el vecino es mayor o igual al centro
            bool isGreaterOrEqual = (neighborValue >= center);
            // Desplaza el bit 1 a la posición correspondiente
            int shiftedBit = isGreaterOrEqual << i;
            // Combina el bit en el código LBP
            lbpCode |= shiftedBit;
        }
    }



    return lbpCode;
}


// ==========================================================================================================================================================================================
// ======================================                                       FIN CUDA V2                                        ==========================================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================

