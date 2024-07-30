package com.example.proyecto_vison;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

public class ImagePickerActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_SELECT_IMAGE = 2;
    private static final int REQUEST_CAMERA_PERMISSION = 100;
    private static final String TAG = "ImagePickerActivity";

    private ImageView imageView;
    private ImageView imageViewHOG;
    private Interpreter tflite;

    static {
        System.loadLibrary("proyecto_vison");
    }

    public native void calcularHOG(long matAddrRgba, long matAddrHOG);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_picker);

        imageView = findViewById(R.id.image_view);
        imageViewHOG = findViewById(R.id.image_view_hog);
        Button btnTakePhoto = findViewById(R.id.btn_take_photo);
        Button btnSelectPhoto = findViewById(R.id.btn_select_photo);

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
        }

        btnTakePhoto.setOnClickListener(v -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
            } else {
                takePhoto();
            }
        });

        btnSelectPhoto.setOnClickListener(v -> selectPhoto());
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("hog_mlp_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                takePhoto();
            } else {
                Toast.makeText(this, "Se necesita permisos de camara", Toast.LENGTH_LONG).show();
            }
        }
    }

    private void takePhoto() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    private void selectPhoto() {
        Intent selectPictureIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(selectPictureIntent, REQUEST_SELECT_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_IMAGE_CAPTURE && data != null) {
                Bundle extras = data.getExtras();
                Bitmap imageBitmap = (Bitmap) extras.get("data");
                imageView.setImageBitmap(imageBitmap);
                enviarImagenAProcesar(imageBitmap);
            } else if (requestCode == REQUEST_SELECT_IMAGE && data != null) {
                Uri imageUri = data.getData();
                try {
                    InputStream imageStream = getContentResolver().openInputStream(imageUri);
                    Bitmap selectedImage = BitmapFactory.decodeStream(imageStream);
                    imageView.setImageBitmap(selectedImage);
                    enviarImagenAProcesar(selectedImage);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void enviarImagenAProcesar(Bitmap bitmapOriginal) {
        Mat matOriginal = new Mat();
        Utils.bitmapToMat(bitmapOriginal, matOriginal);

        // Asegúrate de que la imagen está en formato RGBA antes de convertirla a escala de grises
        if (matOriginal.channels() == 1) {
            Imgproc.cvtColor(matOriginal, matOriginal, Imgproc.COLOR_GRAY2RGBA);
        } else if (matOriginal.channels() == 3) {
            Imgproc.cvtColor(matOriginal, matOriginal, Imgproc.COLOR_BGR2RGBA);
        }

        // Redimensionar la imagen a 28x28 antes de calcular HOG
        Mat resizedImage = new Mat();
        Imgproc.resize(matOriginal, resizedImage, new Size(28, 28));

        Mat matHOG = new Mat();
        calcularHOG(resizedImage.getNativeObjAddr(), matHOG.getNativeObjAddr());

        // Mensajes de depuración
        Log.d(TAG, "Tipo de matHOG: " + matHOG.type());
        Log.d(TAG, "Número de características HOG: " + matHOG.total());

        // Verificar si el tamaño de matHOG es mayor de lo esperado
        int totalSize = (int) (matHOG.total() * matHOG.channels());
        Log.d(TAG, "Tamaño total de matHOG: " + totalSize);

        float[] inputArray = new float[totalSize];
        matHOG.get(0, 0, inputArray);

        // Validar dimensiones de entrada
        int expectedInputSize = tflite.getInputTensor(0).shape()[1]; // Obtener tamaño esperado
        Log.d(TAG, "Tamaño esperado de entrada del modelo: " + expectedInputSize);

        if (totalSize != expectedInputSize) {
            Log.e(TAG, "El tamaño de entrada no coincide: esperado " + expectedInputSize + " pero obtenido " + totalSize);
            return;
        }

        float[][] input = new float[1][totalSize];
        input[0] = inputArray;

        float[][] output = new float[1][10]; // Suponiendo 10 clases de salida

        // Realizar la predicción
        try {
            tflite.run(input, output);
            Log.d(TAG, "Predicción realizada con éxito");
        } catch (Exception e) {
            Log.e(TAG, "Error al ejecutar la predicción: " + e.getMessage());
        }

        // Obtener el resultado de la predicción
        int predictedClass = -1;
        float maxConfidence = -1;
        for (int i = 0; i < output[0].length; i++) {
            if (output[0][i] > maxConfidence) {
                maxConfidence = output[0][i];
                predictedClass = i;
            }
        }

        Toast.makeText(this, "Predicción: " + predictedClass, Toast.LENGTH_LONG).show();

        // Convertir matHOG a Bitmap para visualizar
        Mat matHOG_8UC1 = new Mat();
        matHOG.convertTo(matHOG_8UC1, CvType.CV_8UC1);
        Bitmap bitmapHOG = Bitmap.createBitmap(matHOG_8UC1.cols(), matHOG_8UC1.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matHOG_8UC1, bitmapHOG);
        imageViewHOG.setImageBitmap(bitmapHOG);

        // Liberar memoria de las matrices no necesarias
        matOriginal.release();
        matHOG.release();
        resizedImage.release();
        matHOG_8UC1.release();
    }



}
