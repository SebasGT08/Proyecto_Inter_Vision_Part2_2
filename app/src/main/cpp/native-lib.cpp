#include <jni.h>
#include <opencv2/opencv.hpp>
#include <android/log.h>
#include <vector>

#define LOG_TAG "PROYECTO_VISION"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace cv;
using namespace std;

extern "C" JNIEXPORT void JNICALL
Java_com_example_proyecto_1vison_ImagePickerActivity_calcularHOG(JNIEnv* env, jobject, jlong direccionMatRgba, jlong direccionMatHOG, jlong direccionMatProcessed) {
    Mat& matOriginal = *(Mat*)direccionMatRgba;
    Mat& matHOG = *(Mat*)direccionMatHOG;
    Mat& matProcessed = *(Mat*)direccionMatProcessed;

    // Redimensionar la imagen a 28x28
    Mat resizedImage;
    cv::resize(matOriginal, resizedImage, Size(28, 28));

    // Convertir la imagen a escala de grises solo si tiene 3 o 4 canales
    Mat gray;
    if (resizedImage.channels() == 3) {
        cvtColor(resizedImage, gray, COLOR_BGR2GRAY);
    } else if (resizedImage.channels() == 4) {
        cvtColor(resizedImage, gray, COLOR_RGBA2GRAY);
    } else {
        gray = resizedImage; // Si ya está en escala de grises, úsala directamente
    }

    // Calcular gradientes utilizando el operador Sobel
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 1);
    Sobel(gray, gy, CV_32F, 0, 1, 1);

    // Convertir gradientes a magnitud y ángulo
    Mat magnitud, angulo;
    cartToPolar(gx, gy, magnitud, angulo, true);

    // Configurar los parámetros de HOG para que coincidan con los del entrenamiento
    HOGDescriptor hog(
            Size(28, 28), // winSize
            Size(14, 14), // blockSize
            Size(7, 7),   // blockStride
            Size(7, 7),   // cellSize
            9             // nbins
    );


    // Calcular el descriptor HOG
    vector<float> descriptors;
    hog.compute(gray, descriptors);

    // Imprimir los descriptores HOG
    std::cout << "HOG Features Shape: (1, " << descriptors.size() << ")" << std::endl;
    std::cout << "HOG Features: [";
    for (size_t i = 0; i < descriptors.size(); ++i) {
        std::cout << std::fixed << std::setprecision(8) << descriptors[i];
        if (i < descriptors.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    // Inicializar matHOG con el tamaño y tipo correctos
    matHOG = Mat(descriptors).clone();

    // Procesar la imagen para visualizar (ejemplo: agrandarla para visualizar mejor)
    resize(gray, matProcessed, Size(100, 100), 0, 0, INTER_LINEAR);
}
