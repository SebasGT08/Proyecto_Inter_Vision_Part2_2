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
Java_com_example_proyecto_1vison_ImagePickerActivity_calcularHOG(JNIEnv* env, jobject, jlong direccionMatRgba, jlong direccionMatHOG) {
    Mat& matOriginal = *(Mat*)direccionMatRgba;
    Mat& matHOG = *(Mat*)direccionMatHOG;

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

    // Configurar los parámetros de HOG para que coincidan con los del entrenamiento
    HOGDescriptor hog(
            Size(28, 28), // winSize
            Size(12, 12), // blockSize
            Size(4, 4),   // blockStride
            Size(4, 4),   // cellSize
            9             // nbins
    );

    // Calcular el descriptor HOG
    vector<float> descriptors;
    hog.compute(gray, descriptors);

    // Inicializar matHOG con el tamaño y tipo correctos
    matHOG = Mat(descriptors).clone();
}
