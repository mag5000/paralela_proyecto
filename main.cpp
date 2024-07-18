
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif // STB_IMAGE_IMPLEMENTATION
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif // STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <windows.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void findKeyPointsAndDescriptorsCUDA(const unsigned char* image, int width, int height, int channels, int threshold, int* keypoints, int* descriptors);
__global__ void convertToGrayscaleCUDA(const unsigned char* image, int width, int height, int channels, unsigned char* grayImage);
__global__ void findKeyPointsAndDescriptorsCUDAV2(const unsigned char* image, int width, int height, int channels, int threshold, int* keypoints, int* descriptors);
__global__ void convertToGrayscaleCUDAV2(const unsigned char* image, int width, int height, int channels, unsigned char* grayImage);


// ==========================================================================================================================================================================================
// =================================                                       ALGORITMOS AUXILIARES  P1 EN CPU                                         =========================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================


// Función para guardar puntos clave y descriptores en un archivo CSV
void saveKPToCSV(const char* filename, int* keypoints, int* descriptors, int numKeypoints) {
    // Abrir el archivo para escritura
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error al abrir el archivo CSV para escritura." << std::endl;
        return;
    }

    // Escribir la cabecera del archivo CSV
    file << "x,y,descriptor" << std::endl;

    // Escribir los datos de los puntos clave y descriptores en el archivo CSV
    for (int i = 0; i < numKeypoints; ++i) {
        int x = keypoints[i * 2];
        int y = keypoints[i * 2 + 1];
        int descriptor = descriptors[i];
        file << x << "," << y << "," << descriptor << std::endl;
    }

    // Cerrar el archivo
    file.close();
}

// Funcion para guardar imagen con KP
void saveImageWithKP(char* name, unsigned char* image, int numKeyPoints, int* h_keypoints, int width, int height) {
    // Dibujar los keypoints sobre la imagen

    for (int i = 0; i < numKeyPoints; ++i) {
        int x = h_keypoints[i * 2];
        int y = h_keypoints[i * 2 + 1];
        image[(y * width + x) * 3] = 255;  // R
        image[(y * width + x) * 3 + 1] = 0;  // G
        image[(y * width + x) * 3 + 2] = 0;  // B
    }

    // Guardar la imagen con los keypoints
    stbi_write_jpg(name, width, height, 3, image, 100);

}

// ==========================================================================================================================================================================================
// ======================================                                           FIN AUXILIARES P1                                      ==================================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================



// ==========================================================================================================================================================================================
// ================================                                       ALGORITMOS SECUENCIALES EN CPU                                         ============================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================

// Función para convertir una imagen en escala de grises 
__host__ void convertToGrayscale(const unsigned char* image, int width, int height, int channels, unsigned char* grayImage) {

    for (int row = 0; row < height; ++row) {

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
}

// Función para verificar si un píxel es un punto clave FAST
__host__ int* detectFASTAndReturnKeyPoints(const unsigned char* image, int width, int height, int channels, int threshold, int* numKeyPoints) {
    int maxKeyPoints = width * height; // Definimos el máximo posible de puntos clave
    int* keypoints = new int[maxKeyPoints * 2]; // Almacenamos las coordenadas x e y de los puntos clave
    *numKeyPoints = 0; // Inicializamos el contador de puntos clave

    unsigned char center;
    int pixelOffsets[16][2] = {
        {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}, {0, 3}, {1, 3}, {2, 2}, {3, 1},
        {3, 0}, {3, -1}, {2, -2}, {1, -3}, {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}
    };

    for (int y = 3; y < height - 3; ++y) {
        for (int x = 3; x < width - 3; ++x) {
            center = image[(y * width + x) * channels];
            int brighterCount = 0;
            int darkerCount = 0;

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
                keypoints[*numKeyPoints * 2] = x;
                keypoints[*numKeyPoints * 2 + 1] = y;
                (*numKeyPoints)++;
            }
        }
    }
    return keypoints;
}

// Función para calcular el LBP para un punto en una imagen en escala de grises con un parche de 5x5
__host__ int calculateLBPCPU(const unsigned char* image, int width, int x, int y) {
    int center = image[y * width + x];
    int lbpCode = 0;

    // Vecinos en sentido horario empezando desde el de arriba
    int neighborOffsets[24][2] = {
        {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
        {2, -1}, {2, 0}, {2, 1}, {2, 2},
        {1, 2}, {0, 2}, {-1, 2}, {-2, 2},
        {-2, 1}, {-2, 0}, {-2, -1},
        {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}
    };

    for (int i = 0; i < 24; ++i) {
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

// Función para calcular el LBP para cada punto clave detectado por el algoritmo FAST
__host__ int* calculateLBPDescriptorsCPU(const unsigned char* image, int width, int height, const int* keypoints, int numKeyPoints) {
    int* lbpDescriptors = new int[numKeyPoints];

    for (int i = 0; i < numKeyPoints; ++i) {
        int x = keypoints[i * 2];
        int y = keypoints[i * 2 + 1];
        int lbpCode = calculateLBPCPU(image, width, x, y);
        lbpDescriptors[i] = lbpCode;
    }

    return lbpDescriptors;
}

__host__ void callKPFinderCPU(char* image_name, int threshold) {

    // Cargar la imagen
    int width, height, channels;
    string path = "Dataset/";
    unsigned char* raw_image = stbi_load(path.append(image_name).c_str(), &width, &height, &channels, 0);
    unsigned char* image = new unsigned char[width * height];

    if (!raw_image) {
        cerr << "Error al cargar la imagen." << endl;
    }

    if (channels != 1) {
        convertToGrayscale(raw_image, width, height, channels, image);
        channels = 1;
    }
    else {
        image = raw_image;
    }

    // Comentar aqui y linea 424 (aprox) para guardar imagen RGB con KP 
    // delete[] raw_image;

    // Copiar los puntos clave y descriptores de la GPU a la CPU
    int numKeyPoints = 0;
    int* keypoints = detectFASTAndReturnKeyPoints(image, width, height, channels, threshold, &numKeyPoints);
    int* descriptors = calculateLBPDescriptorsCPU(image, width, height, keypoints, numKeyPoints);

    // Descomentar para guardar imagen RGB con KP (Descomentar tambien linea 410 aprox) 
    saveImageWithKP("Output/imagen_con_keypoints_CPU.jpg", raw_image, numKeyPoints, keypoints, width, height);

    // Descomentar para crear archivo con puntos claves y descriptor
    // saveKPToCSV("keypoints_CPU.csv", keypoints, descriptors, numKeyPoints);

    // Liberar la memoria de los keypoints y descriptores
    delete[] keypoints;
    delete[] descriptors;
    delete[] image;

}


__host__ void callKPFinderCPUTimes(int exp_x_array, char* image_name, int threshold) {

    cout << "Analizando: " << image_name << endl;

    long long int total_load = 0;
    long long int total_grayscale = 0;
    long long int total_keypoints = 0;
    long long int total_descriptors = 0;

    long long int avg_load = 0;
    long long int avg_grayscale = 0;
    long long int avg_keypoints = 0;
    long long int avg_descriptors = 0;

    // Verificar si el archivo ya existe
    std::ifstream infile("Output/secuencial_times.csv");
    bool archivo_existe = infile.good();
    infile.close();

    // Abre el archivo en modo de añadir al final
    std::ofstream outfile("Output/secuencial_times.csv", ofstream::out | ofstream::app);
    // Verifica si el archivo se abrió correctamente
    if (!outfile) {
        cout << "Error al abrir el archivo." << endl;
    }
    // Escribir el encabezado solo si el archivo no existía previamente
    if (!archivo_existe) {
        outfile << "Algoritmo,Imagen,Tiempo\n";
    }

    for (int i = 0; i < exp_x_array; i++) {

        // Set tiempo inicio 
        auto start_load = chrono::steady_clock::now();

        // Cargar la imagen
        int width, height, channels;
        string path = "Dataset/";
        unsigned char* raw_image = stbi_load(path.append(image_name).c_str(), &width, &height, &channels, 0);
        unsigned char* image = new unsigned char[width * height];
        // Esta ok?
        if (!raw_image) {
            cerr << "Error al cargar la imagen." << endl;
        }
        // Tiempo fin carga
        auto end_load = chrono::steady_clock::now();


        // Set tiempo inicio 
        auto start_grayscale = chrono::steady_clock::now();

        convertToGrayscale(raw_image, width, height, channels, image);
        channels = 1;
        // Limpiamos memoria
        delete[] raw_image;

        // Tiempo fin grayscale
        auto end_grayscale = chrono::steady_clock::now();


        // Set tiempo inicio 
        // Inicializamos keypoints
        int numKeyPoints = 0;
        auto start_keypoints = chrono::steady_clock::now();
        // Buscamos keypoints
        int* keypoints = detectFASTAndReturnKeyPoints(image, width, height, channels, threshold, &numKeyPoints);
        // Tiempo fin keypoints
        auto end_keypoints = chrono::steady_clock::now();

        // Set tiempo inicio 
        auto start_descriptors = chrono::steady_clock::now();
        // calculamos descriptores
        int* descriptors = calculateLBPDescriptorsCPU(image, width, height, keypoints, numKeyPoints);
        // Tiempo fin descriptores
        auto end_descriptors = chrono::steady_clock::now();

        long long load_time = chrono::duration_cast<std::chrono::microseconds>(end_load - start_load).count();
        long long grayscale_time = chrono::duration_cast<std::chrono::microseconds>(end_grayscale - start_grayscale).count();
        long long keypoints_time = chrono::duration_cast<std::chrono::microseconds>(end_keypoints - start_keypoints).count();
        long long descriptors_time = chrono::duration_cast<std::chrono::microseconds>(end_descriptors - start_descriptors).count();

        total_load = total_load + load_time;
        total_grayscale = total_grayscale + grayscale_time;
        total_keypoints = total_keypoints + keypoints_time;
        total_descriptors = total_descriptors + descriptors_time;

        delete[] keypoints;
        delete[] descriptors;
        delete[] image;

    }


    avg_load = total_load / exp_x_array;
    avg_grayscale = total_grayscale / exp_x_array;
    avg_keypoints = total_keypoints / exp_x_array;
    avg_descriptors = total_descriptors / exp_x_array;

    outfile << "Load," << image_name << "," << avg_load << endl;
    outfile << "grayscale," << image_name << "," << avg_grayscale << endl;
    outfile << "keypoints," << image_name << "," << avg_keypoints << endl;
    outfile << "descriptors," << image_name << "," << avg_descriptors << endl;


}


// ==========================================================================================================================================================================================
// ======================================                                FIN CLASICOS SECUENCIALES CPU                                      =================================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================


// ==========================================================================================================================================================================================
// ======================================                                ALGORITMOS QUE LLAMAN CUDA                                       ===================================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================

// ================================================= GPU V1 =========================================

// Función en CPU que llama a GPU V1
__host__ void callKPFinderCUDA(char* image_name, int threshold) {

    // Cargar la imagen
    int width, height, channels;
    string path = "Dataset/";
    unsigned char* raw_image = stbi_load(path.append(image_name).c_str(), &width, &height, &channels, 0);
    //unsigned char* image = new unsigned char[width * height]; // solo es una espacio de tamaño de la imagen en escala de grises

    if (!raw_image) {
        cerr << "Error al cargar la imagen." << endl;
    }

    // Tamaño de la imagen en bytes
    size_t RawImageSize = width * height * channels * sizeof(unsigned char);
    size_t imageSize = width * height * sizeof(unsigned char);

    // Espacio para imagen en GPU
    unsigned char* d_image;
    cudaMalloc((void**)&d_image, imageSize);

    // Configurar la grilla y los bloques de hilos
    int blockSize = 128; // Threads por bloque
    int numBlocks = (height + blockSize - 1) / blockSize; // El numero de bloques mínimo aproximado

    // Si la imagen no esta en escala de grises la convertimos en la GPU
    if (channels != 1) {

        // Copiar la imagen a la GPU
        unsigned char* d_raw_image;
        cudaMalloc((void**)&d_raw_image, RawImageSize);
        cudaMemcpy(d_raw_image, raw_image, RawImageSize, cudaMemcpyHostToDevice);
        //cudaMemcpy(d_image, 0, imageSize, cudaMemcpyHostToDevice);

        // Convertir imagen
        convertToGrayscaleCUDA << <numBlocks, blockSize >> > (d_raw_image, width, height, channels, d_image);
        cudaDeviceSynchronize();
        cudaError_t err;
        err = cudaGetLastError();  // Obtener el último error después de lanzar el kernel
        if (err != cudaSuccess) {
            fprintf(stderr, "Error en el kernel CUDA GrayScale: %s\n", cudaGetErrorString(err));
            // Manejo de errores adecuado aquí
        }
        channels = 1;



        //borramos la imagen raw de memoria
        cudaFree(d_raw_image);

        //EN caso que la imagen ya este en escala de grises solo la llevamos a la GPU
    }
    else {

        // Copiar la imagen a la GPU   
        cudaMemcpy(d_image, raw_image, imageSize, cudaMemcpyHostToDevice);

    }

    // Comentar aqui y linea 230 (aprox) para guardar imagen RGB con KP 
    // delete[] raw_image;

    // Crear un vector para almacenar los puntos clave
    int initialCount = 0;
    int maxKeyPoints = width * height; // Supongamos el máximo número de puntos clave posible
    int keyPointMatrixSize = maxKeyPoints * 2 * sizeof(int);
    int* d_keypoints;
    cudaMalloc((void**)&d_keypoints, keyPointMatrixSize);
    cudaMemcpy(d_keypoints, &initialCount, sizeof(int), cudaMemcpyHostToDevice); // Inicializar el contador en la GPU

    // Crear una matriz para almacenar los descriptores LBP
    int* d_descriptors;
    cudaMalloc((void**)&d_descriptors, maxKeyPoints * sizeof(int));

    // Detectar puntos clave FAST y calcular descriptores en paralelo en la GPU
    findKeyPointsAndDescriptorsCUDA << <numBlocks, blockSize >> > (d_image, width, height, channels, threshold, d_keypoints, d_descriptors);
    cudaDeviceSynchronize();
    cudaError_t err;
    err = cudaGetLastError();  // Obtener el último error después de lanzar el kernel
    if (err != cudaSuccess) {
        fprintf(stderr, "Error en el kernel CUDA FindKP: %s\n", cudaGetErrorString(err));
        // Manejo de errores adecuado aquí
    }

    // Copiar los puntos clave y descriptores de la GPU a la CPU
    int numKeyPoints;
    cudaMemcpy(&numKeyPoints, d_keypoints, sizeof(int), cudaMemcpyDeviceToHost);
    int* h_keypoints = new int[numKeyPoints * 2];
    cudaMemcpy(h_keypoints, d_keypoints + 1, numKeyPoints * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int* h_descriptors = new int[numKeyPoints];
    cudaMemcpy(h_descriptors, d_descriptors, numKeyPoints * sizeof(int), cudaMemcpyDeviceToHost);

    // Descomentar para guardar imagen RGB con KP (Descomentar tambien linea 180 aprox) 
    saveImageWithKP("Output/imagen_con_keypoints_CUDA.jpg", raw_image, numKeyPoints, h_keypoints, width, height);
    // Descomentar para crear archivo con puntos claves y descriptor
    // saveKPToCSV("Keypoints_CUDA.csv", h_keypoints, h_descriptors, numKeyPoints);

    // Liberar la memoria de los keypoints y descriptores e imagenes
    delete[] h_keypoints;
    delete[] h_descriptors;
    cudaFree(d_keypoints);
    cudaFree(d_descriptors);
    cudaFree(d_image);

}

// ================================================= CPU Y GPU =========================================

// Función en CPU que llama a GPU (CPU + GPU)
__host__ void callKPFinderCUDACPU(char* image_name, int threshold) {

    // Cargar la imagen
    int width, height, channels;
    string path = "Dataset/";
    unsigned char* raw_image = stbi_load(path.append(image_name).c_str(), &width, &height, &channels, 0);
    unsigned char* image = new unsigned char[width * height]; // solo es una espacio de tamaño de la imagen en escala de grises

    if (!raw_image) {
        cerr << "Error al cargar la imagen." << endl;
    }

    // Si la imagen no esta en escala de grises la cinvertimos en la CPU
    if (channels != 1) {
        convertToGrayscale(raw_image, width, height, channels, image);
        channels = 1;
    }
    else {

        image = raw_image;

    }

    // Comentar aqui y Descomentar linea 520 (aprox) para guardar imagen RGB con KP 
    // delete[] raw_image;

    // Imagen para GPU
    size_t imageSize = width * height * sizeof(unsigned char);
    unsigned char* d_image;
    cudaMalloc((void**)&d_image, imageSize);
    cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);

    // Crear un vector para almacenar los puntos clave
    int initialCount = 0;
    int maxKeyPoints = width * height; // Supongamos el máximo número de puntos clave posible
    int keyPointMatrixSize = maxKeyPoints * 2 * sizeof(int);
    int* d_keypoints;
    cudaMalloc((void**)&d_keypoints, keyPointMatrixSize);
    cudaMemcpy(d_keypoints, &initialCount, sizeof(int), cudaMemcpyHostToDevice); // Inicializar el contador en la GPU

    // Crear una matriz para almacenar los descriptores LBP
    int* d_descriptors;
    cudaMalloc((void**)&d_descriptors, maxKeyPoints * sizeof(int));

    // Configurar la grilla y los bloques de hilos
    int blockSize = 128; // Threads por bloque
    int numBlocks = (height + blockSize - 1) / blockSize; // El numero de bloques mínimo aproximado

    // Detectar puntos clave FAST y calcular descriptores en paralelo en la GPU
    findKeyPointsAndDescriptorsCUDA << <numBlocks, blockSize >> > (d_image, width, height, channels, threshold, d_keypoints, d_descriptors);
    cudaError_t err;
    cudaDeviceSynchronize();
    err = cudaGetLastError();  // Obtener el último error después de lanzar el kernel
    if (err != cudaSuccess) {
        fprintf(stderr, "Error en el kernel CUDACPU FindKP: %s\n", cudaGetErrorString(err));
        // Manejo de errores adecuado aquí
    }

    // Copiar los puntos clave y descriptores de la GPU a la CPU
    int numKeyPoints;
    cudaMemcpy(&numKeyPoints, d_keypoints, sizeof(int), cudaMemcpyDeviceToHost);
    int* h_keypoints = new int[numKeyPoints * 2];
    cudaMemcpy(h_keypoints, d_keypoints + 1, numKeyPoints * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int* h_descriptors = new int[numKeyPoints];
    cudaMemcpy(h_descriptors, d_descriptors, numKeyPoints * sizeof(int), cudaMemcpyDeviceToHost);

    // Descomentar para guardar imagen RGB con KP (Descomentar tambien linea 490 aprox) 
    saveImageWithKP("Output/imagen_con_keypoints_CUDAGPU.jpg", raw_image, numKeyPoints, h_keypoints, width, height);

    // Descomentar para guardar imagen en grayscale
    // stbi_write_jpg("Gray_image_CUDA_CPU.jpg", width, height, 1, image, 100);
    // stbi_image_free(raw_image);

    // Descomentar para crear archivo con puntos claves y descriptor
    //  saveKPToCSV("Keypoints_CUDA_CPU.csv", h_keypoints, h_descriptors, numKeyPoints);

    // Liberar la memoria de los keypoints y descriptores
    delete[] h_keypoints;
    delete[] h_descriptors;
    cudaFree(d_keypoints);
    cudaFree(d_descriptors);
    cudaFree(d_image);

}


// ================================================= GPU V2 =========================================


// Función en CPU que llama a GPU V2
__host__ void callKPFinderCUDAV2(char* image_name, int threshold) {

    // Cargar la imagen
    int width, height, channels;
    string path = "Dataset/";
    unsigned char* raw_image = stbi_load(path.append(image_name).c_str(), &width, &height, &channels, 0);
    unsigned char* image = new unsigned char[width * height]; // solo es una espacio de tamaño de la imagen en escala de grises

    // Tamaño de la imagen en bytes
    size_t RawImageSize = width * height * channels * sizeof(unsigned char);
    size_t imageSize = width * height * sizeof(unsigned char);

    // Espacio para imagen en GPU
    unsigned char* d_image;
    cudaMalloc((void**)&d_image, imageSize);

    // Configuración de tamaño de bloque y grilla para el kernel (ejemplo)
    dim3 blockSize(16, 16); // Tamaño del bloque 16x16 threads
    dim3 gridSize;

    // Configurar la grilla para el kernel
    gridSize.x = (width + blockSize.x - 1) / blockSize.x;
    gridSize.y = (height + blockSize.y - 1) / blockSize.y;

    // Si la imagen no esta en escala de grises la convertimos en la GPU
    if (channels != 1) {

        // Copiar la imagen a la GPU
        unsigned char* d_raw_image;
        cudaMalloc((void**)&d_raw_image, RawImageSize);
        cudaMemcpy(d_raw_image, raw_image, RawImageSize, cudaMemcpyHostToDevice);

        // Convertir imagen
        convertToGrayscaleCUDAV2 << <gridSize, blockSize >> > (d_raw_image, width, height, channels, d_image);
        cudaError_t err;
        cudaDeviceSynchronize();
        err = cudaGetLastError();  // Obtener el último error después de lanzar el kernel
        if (err != cudaSuccess) {
            fprintf(stderr, "Error en el kernel CUDA GrayScale: %s\n", cudaGetErrorString(err));
            // Manejo de errores adecuado aquí
        }
        channels = 1;

        //borramos la imagen raw de memoria de GPU
        cudaFree(d_raw_image);

        //EN caso que la imagen ya este en escala de grises solo la llevamos a la GPU
    }
    else {

        // Copiar la imagen a la GPU   
        cudaMemcpy(d_image, raw_image, imageSize, cudaMemcpyHostToDevice);

    }

    // Borramos la imagen de memoria
    // Comentar aqui y Descomentar linea 520 (aprox) para guardar imagen RGB con KP 
    // delete[] raw_image;

    // Crear un vector para almacenar los puntos clave
    int initialCount = 0;
    int maxKeyPoints = width * height; // Supongamos el máximo número de puntos clave posible
    int keyPointMatrixSize = maxKeyPoints * 2 * sizeof(int);
    int* d_keypoints;
    cudaMalloc((void**)&d_keypoints, keyPointMatrixSize);
    cudaMemcpy(d_keypoints, &initialCount, sizeof(int), cudaMemcpyHostToDevice); // Inicializar el contador en la GPU

    // Crear una matriz para almacenar los descriptores LBP
    int* d_descriptors;
    cudaMalloc((void**)&d_descriptors, maxKeyPoints * sizeof(int));

    // Detectar puntos clave FAST y calcular descriptores en paralelo en la GPU
    findKeyPointsAndDescriptorsCUDAV2 << <gridSize, blockSize >> > (d_image, width, height, channels, threshold, d_keypoints, d_descriptors);
    cudaError_t err;
    cudaDeviceSynchronize();
    err = cudaGetLastError();  // Obtener el último error después de lanzar el kernel
    if (err != cudaSuccess) {
        fprintf(stderr, "Error en el kernel CUDACPU FindKP: %s\n", cudaGetErrorString(err));
        // Manejo de errores adecuado aquí
    }

    // Copiar los puntos clave y descriptores de la GPU a la CPU
    int numKeyPoints;
    cudaMemcpy(&numKeyPoints, d_keypoints, sizeof(int), cudaMemcpyDeviceToHost);
    int* h_keypoints = new int[numKeyPoints * 2];
    cudaMemcpy(h_keypoints, d_keypoints + 1, numKeyPoints * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int* h_descriptors = new int[numKeyPoints];
    cudaMemcpy(h_descriptors, d_descriptors, numKeyPoints * sizeof(int), cudaMemcpyDeviceToHost);

    // Descomentar para guardar imagen RGB con KP (Descomentar tambien linea 490 aprox) 
    saveImageWithKP("Output/imagen_con_keypoints_CUDAV2.jpg", raw_image, numKeyPoints, h_keypoints, width, height);
    // Descomentar para crear archivo con puntos claves y descriptor
    // saveKPToCSV("Keypoints_CUDAV2.csv", h_keypoints, h_descriptors, numKeyPoints);

    // Liberar la memoria de los keypoints y descriptores
    delete[] h_keypoints;
    delete[] h_descriptors;
    cudaFree(d_keypoints);
    cudaFree(d_descriptors);
    cudaFree(d_image);

}


// ================================================= GPU y CPU V2 =========================================
// Función en CPU que llama a GPU (CPU y GPU V2)
__host__ void callKPFinderCUDACPUV2(char* image_name, int threshold) {

    // Cargar la imagen
    int width, height, channels;
    string path = "Dataset/";
    unsigned char* raw_image = stbi_load(path.append(image_name).c_str(), &width, &height, &channels, 0);
    unsigned char* image = new unsigned char[width * height]; // solo es una espacio de tamaño de la imagen en escala de grises


    if (!raw_image) {
        cerr << "Error al cargar la imagen." << endl;
    }

    // Si la imagen no esta en escala de grises la cinvertimos en la CPU
    if (channels != 1) {
        convertToGrayscale(raw_image, width, height, channels, image);
        channels = 1;
    }
    else {

        image = raw_image;

    }

    // Borramos la imagen de memoria
    // Comentar aqui y Descomentar linea 520 (aprox) para guardar imagen RGB con KP 
    // delete[] raw_image;

    // Movemos la imagen a la GPU
    size_t imageSize = width * height * sizeof(unsigned char);
    unsigned char* d_image;
    cudaMalloc((void**)&d_image, imageSize);
    cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);

    // Crear un vector para almacenar los puntos clave
    int initialCount = 0;
    int maxKeyPoints = width * height; // Supongamos el máximo número de puntos clave posible
    int keyPointMatrixSize = maxKeyPoints * 2 * sizeof(int);
    int* d_keypoints;
    cudaMalloc((void**)&d_keypoints, keyPointMatrixSize);
    cudaMemcpy(d_keypoints, &initialCount, sizeof(int), cudaMemcpyHostToDevice); // Inicializar el contador en la GPU

    // Crear una matriz para almacenar los descriptores LBP
    int* d_descriptors;
    cudaMalloc((void**)&d_descriptors, maxKeyPoints * sizeof(int));

    // Configuración de tamaño de bloque y grilla para el kernel (ejemplo)
    dim3 blockSize(16, 16); // Tamaño del bloque 16x16 threads
    dim3 gridSize;
    // Configurar la grilla para el kernel
    gridSize.x = (width + blockSize.x - 1) / blockSize.x;
    gridSize.y = (height + blockSize.y - 1) / blockSize.y;

    // Detectar puntos clave FAST y calcular descriptores en paralelo en la GPU
    findKeyPointsAndDescriptorsCUDAV2 << <gridSize, blockSize >> > (d_image, width, height, channels, threshold, d_keypoints, d_descriptors);
    cudaError_t err;
    cudaDeviceSynchronize();
    err = cudaGetLastError();  // Obtener el último error después de lanzar el kernel
    if (err != cudaSuccess) {
        fprintf(stderr, "Error en el kernel CUDACPU FindKP: %s\n", cudaGetErrorString(err));
        // Manejo de errores adecuado aquí
    }

    // Copiar los puntos clave y descriptores de la GPU a la CPU
    int numKeyPoints;
    cudaMemcpy(&numKeyPoints, d_keypoints, sizeof(int), cudaMemcpyDeviceToHost);
    int* h_keypoints = new int[numKeyPoints * 2];
    cudaMemcpy(h_keypoints, d_keypoints + 1, numKeyPoints * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int* h_descriptors = new int[numKeyPoints];
    cudaMemcpy(h_descriptors, d_descriptors, numKeyPoints * sizeof(int), cudaMemcpyDeviceToHost);

    // Descomentar para guardar imagen RGB con KP (Descomentar tambien linea 490 aprox) 
    saveImageWithKP("Output/imagen_con_keypoints_CUDAGPUV2.jpg", raw_image, numKeyPoints, h_keypoints, width, height);
    // Descomentar para crear archivo con puntos claves y descriptor
    // saveKPToCSV("Keypoints_CUDACPUV2.csv", h_keypoints, h_descriptors, numKeyPoints);

    // Liberar la memoria de los keypoints y descriptores
    delete[] h_keypoints;
    delete[] h_descriptors;
    cudaFree(d_keypoints);
    cudaFree(d_descriptors);
    cudaFree(d_image);

}


// ==========================================================================================================================================================================================
// ======================================                              FIN ALGORITMOS QUE LLAMAN CUDA                                     ===================================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================


// ==========================================================================================================================================================================================
// =================================                                       ALGORITMOS AUXILIARES P2                                      =============================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================


void takeTime(int exp_x_array, char* image_name, int threshold) {

    long long int avg_CPU_time = 0;
    long long int avg_CUDA_time = 0;
    long long int avg_CUDACPU_time = 0;
    long long int total_CPU_time = 0;
    long long int total_CUDA_time = 0;
    long long int total_CUDACPU_time = 0;

    long long int total_CUDAV2_time = 0;
    long long int avg_CUDAV2_time = 0;

    long long int total_CUDACPUV2_time = 0;
    long long int avg_CUDACPUV2_time = 0;

    // Verificar si el archivo ya existe
    std::ifstream infile("Output/results.csv");
    bool archivo_existe = infile.good();
    infile.close();

    // Abre el archivo en modo de añadir al final
    std::ofstream outfile("Output/results.csv", ofstream::out | ofstream::app);
    // Verifica si el archivo se abrió correctamente
    if (!outfile) {
        cout << "Error al abrir el archivo." << endl;
    }
    // Escribir el encabezado solo si el archivo no existía previamente
    if (!archivo_existe) {
        outfile << "Algoritmo,Imagen,Tiempo\n";
    }

    cout << image_name << "    CUDA+CPU:";
    for (int i = 0; i < exp_x_array; i++) {

        auto start_time = chrono::steady_clock::now();

        callKPFinderCUDACPU(image_name, threshold);

        auto end_time = chrono::steady_clock::now();
        long long duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        total_CUDACPU_time = total_CUDACPU_time + duration;
    }

    avg_CUDACPU_time = total_CUDACPU_time / exp_x_array;
    outfile << "CUDA CPU," << image_name << "," << avg_CUDACPU_time << endl;
    cout << "OK | CPU:";
    for (int i = 0; i < exp_x_array; i++) {

        auto start_time = chrono::steady_clock::now();

        callKPFinderCPU(image_name, threshold);

        auto end_time = chrono::steady_clock::now();
        long long duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        total_CPU_time = total_CPU_time + duration;

    }

    avg_CPU_time = total_CPU_time / exp_x_array;
    outfile << "CPU," << image_name << "," << avg_CPU_time << endl;

    cout << "OK | CUDA:";
    for (int i = 0; i < exp_x_array; i++) {

        auto start_time = chrono::steady_clock::now();

        callKPFinderCUDA(image_name, threshold);

        auto end_time = chrono::steady_clock::now();
        long long duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        total_CUDA_time = total_CUDA_time + duration;

    }

    avg_CUDA_time = total_CUDA_time / exp_x_array;
    outfile << "CUDA, " << image_name << "," << avg_CUDA_time << endl;


    cout << "OK | CUDA V2:";
    for (int i = 0; i < exp_x_array; i++) {

        auto start_time = chrono::steady_clock::now();

        callKPFinderCUDAV2(image_name, threshold);

        auto end_time = chrono::steady_clock::now();
        long long duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        total_CUDAV2_time = total_CUDAV2_time + duration;

    }

    avg_CUDAV2_time = total_CUDAV2_time / exp_x_array;
    outfile << "CUDA V2, " << image_name << "," << avg_CUDAV2_time << endl;


    cout << "OK | CUDA CPU V2:";
    for (int i = 0; i < exp_x_array; i++) {

        auto start_time = chrono::steady_clock::now();

        callKPFinderCUDACPUV2(image_name, threshold);

        auto end_time = chrono::steady_clock::now();
        long long duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        total_CUDACPUV2_time = total_CUDACPUV2_time + duration;

    }

    avg_CUDACPUV2_time = total_CUDACPUV2_time / exp_x_array;
    outfile << "CUDA CPU V2, " << image_name << "," << avg_CUDACPUV2_time << endl;
    cout << "OK" << endl;
    outfile.close();
}

// ==========================================================================================================================================================================================
// ======================================                                           FIN AUXILIARES P2                                    ====================================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================



// ==========================================================================================================================================================================================
// ======================================                                                  MAIN                                          ====================================================
// ==========================================================================================================================================================================================
// ==========================================================================================================================================================================================

// Función Main
int main() {

    int threshold = 25;              // Parámetros del algoritmo FAST

    WIN32_FIND_DATA findFileData;
    HANDLE hFind;
    const char* directoryPath = "Dataset\\*"; // Ruta al directorio

    hFind = FindFirstFile(directoryPath, &findFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                // cout << findFileData.cFileName << endl;
                takeTime(1, findFileData.cFileName, threshold);
                // callKPFinderCPUTimes(10, findFileData.cFileName, threshold);

            }
        } while (FindNextFile(hFind, &findFileData));
        FindClose(hFind);
    }
    else {
        cerr << "Error: No se pudo abrir el directorio.\n";
    }

    return 0;
}

