package com.example.proyecto_vison;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.util.Collections;
import java.util.List;

import io.socket.client.IO;
import io.socket.client.Socket;
import io.socket.emitter.Emitter;
import okio.ByteString;

import android.content.Intent;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";
    private static final int CODIGO_SOLICITUD_PERMISO_CAMARA = 200;
    private CameraBridgeViewBase vistaCamara;
    private Mat frameOriginal;
    private long tiempoUltimoFrame = 0;
    private static final int INTERVALO_FRAME = 150; // Intervalo entre cuadros en ms (10 fps)
    private boolean esCamaraFrontal = false; // Para controlar la cámara
    private boolean esModoDibujar = true; // Modo de dibujo inicial

    static {

        System.loadLibrary("proyecto_vison");
    }



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate llamado");
        super.onCreate(savedInstanceState);

        // Inicializar OpenCV
        if (OpenCVLoader.initDebug()) {
            Log.i(TAG, "OpenCV cargado exitosamente");
        } else {
            Log.e(TAG, "¡Fallo en la inicialización de OpenCV!");
            Toast.makeText(this, "¡Fallo en la inicialización de OpenCV!", Toast.LENGTH_LONG).show();
            return;
        }

        // Mantener la pantalla encendida mientras la actividad está en ejecución
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        // Verificar y solicitar permiso para la cámara si no está otorgado
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CODIGO_SOLICITUD_PERMISO_CAMARA);
        }



        // Configurar botón para cambiar el modo de dibujo
        Button botonCambiarModoDibujo = findViewById(R.id.switch_draw_mode_button);
        botonCambiarModoDibujo.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, ImagePickerActivity.class);
            startActivity(intent);
        });



    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == CODIGO_SOLICITUD_PERMISO_CAMARA) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                inicializarCamara();
            } else {
                Toast.makeText(this, "Se requiere permiso para la cámara", Toast.LENGTH_LONG).show();
            }
        }
    }

    // Inicializar vista de la cámara
    private void inicializarCamara() {
        vistaCamara.setVisibility(SurfaceView.VISIBLE);
        vistaCamara.enableView();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (vistaCamara != null) {
            vistaCamara.disableView();
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (vistaCamara != null && vistaCamara.getVisibility() == SurfaceView.VISIBLE) {
            vistaCamara.enableView();
        }
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(vistaCamara);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (vistaCamara != null) {
            vistaCamara.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        frameOriginal = new Mat(height, width, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        if (frameOriginal != null) {
            frameOriginal.release();
        }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {


        return null;
    }
}
