package com.example.fashionai;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private final int INPUT_IMAGE_SIZE = 256,
            PHOTO_FROM_CAMERA_REQUEST_CODE = 0,
            PHOTO_FROM_GALLERY_REQUEST_CODE = 1,
            PERMISSION_REQUEST_CODE = 2,
            IMAGE_VIEW_COUNT = 4;
    private final String MODEL_FILE_NAME = "model_256.tflite";
    ImageView[] imageViewOutputs, imageViewInputs;
    Interpreter tfLite;
    Mat sourceMat, grayMat, invertColorMatrix, invertedGrayMat, blurredMat, invertedBlurredMat, pencilSketchMat, pencilSketchRGBMat;
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    sourceMat = new Mat();
                    grayMat = new Mat();
                    invertedGrayMat = new Mat(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, CvType.CV_8UC1);
                    blurredMat = new Mat();
                    invertedBlurredMat = new Mat(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, CvType.CV_8UC1);
                    pencilSketchMat = new Mat(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, CvType.CV_8UC1);
                    pencilSketchRGBMat = new Mat();
                    invertColorMatrix = new Mat(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, CvType.CV_8UC1, new Scalar(255,255,255));
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button btnCamera = (Button) findViewById(R.id.btnCamera);
        Button btnGallery = (Button) findViewById(R.id.btnGallery);

        imageViewInputs = new ImageView[IMAGE_VIEW_COUNT];
        imageViewInputs[0] = (ImageView) findViewById(R.id.imageViewInput0);
        imageViewInputs[1] = (ImageView) findViewById(R.id.imageViewInput1);
        imageViewInputs[2] = (ImageView) findViewById(R.id.imageViewInput2);
        imageViewInputs[3] = (ImageView) findViewById(R.id.imageViewInput3);
        imageViewOutputs = new ImageView[IMAGE_VIEW_COUNT];
        imageViewOutputs[0] = (ImageView) findViewById(R.id.imageViewOutput0);
        imageViewOutputs[1] = (ImageView) findViewById(R.id.imageViewOutput1);
        imageViewOutputs[2] = (ImageView) findViewById(R.id.imageViewOutput2);
        imageViewOutputs[3] = (ImageView) findViewById(R.id.imageViewOutput3);

        try {
            tfLite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
        }

        btnCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, PHOTO_FROM_CAMERA_REQUEST_CODE);
            }
        });

        btnGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_PICK,android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, PHOTO_FROM_GALLERY_REQUEST_CODE);
            }
        });

        requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
    }

    private ByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(MODEL_FILE_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void onResume(){
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        Bitmap inputBitmap = null;

        switch (requestCode){
            case PHOTO_FROM_CAMERA_REQUEST_CODE:
                inputBitmap = (Bitmap) data.getExtras().get("data");
                break;

            case PHOTO_FROM_GALLERY_REQUEST_CODE:
                if(resultCode != RESULT_OK || data == null) return;

                Uri selectedImage =  data.getData();
                if(selectedImage == null) return;

                String[] filePathColumn = {MediaStore.Images.Media.DATA};

                Cursor cursor = getContentResolver().query(selectedImage,
                        filePathColumn, null, null, null);
                if(cursor == null) return;

                cursor.moveToFirst();

                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                String picturePath = cursor.getString(columnIndex);
                File pictureFile = new File(picturePath);
                inputBitmap = BitmapFactory.decodeFile(pictureFile.getAbsolutePath());
                cursor.close();
                break;

            default:
                return;
        }

        try{
            List<Bitmap> inputBitMaps = getSketchBitmaps(inputBitmap);
            for(int i=0; i<IMAGE_VIEW_COUNT; i++) {
                imageViewInputs[i].setImageBitmap(inputBitMaps.get(i));
                Bitmap outputBitmap = doInterference(inputBitMaps.get(i));
                imageViewOutputs[i].setImageBitmap(outputBitmap);
            }
        }catch (Exception e){
            showAlert(e.getMessage());
        }
    }

    private void showAlert(String msg) {
        new AlertDialog.Builder(this)
                .setTitle("Error")
                .setMessage(msg)
                .setIcon(android.R.drawable.ic_dialog_alert)
                .show();
    }

    private Bitmap doInterference(Bitmap inputBitmap) {
        float[][][][] inputArray = bitmapToFloatArray(inputBitmap);
        float[][][][] outputArray = new float[1][INPUT_IMAGE_SIZE][INPUT_IMAGE_SIZE][3];
        tfLite.run(inputArray, outputArray);
        return floatArrayToBitmap(outputArray);
    }

    private List<Bitmap> getSketchBitmaps(Bitmap bitmap) {
        Bitmap bitmapScaled = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);
        Utils.bitmapToMat(bitmapScaled, sourceMat);

        Imgproc.cvtColor(sourceMat, grayMat, Imgproc.COLOR_RGB2GRAY);
        Core.subtract(invertColorMatrix, grayMat, invertedGrayMat);
        Imgproc.GaussianBlur(invertedGrayMat, blurredMat, new Size(21, 21), 0);
        Core.subtract(invertColorMatrix, blurredMat, invertedBlurredMat);
        Core.divide(grayMat, invertedBlurredMat, pencilSketchMat);

        Imgproc.cvtColor(pencilSketchMat, pencilSketchRGBMat, Imgproc.COLOR_GRAY2RGB);

        Bitmap originalBitmap = Bitmap.createBitmap(sourceMat.cols(), sourceMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(sourceMat, originalBitmap);

        Bitmap pencilSketchRGBBitmap = Bitmap.createBitmap(pencilSketchRGBMat.cols(), pencilSketchRGBMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(pencilSketchRGBMat, pencilSketchRGBBitmap);

        List<Bitmap> sketchedBitmaps = new ArrayList<>();
        sketchedBitmaps.add(originalBitmap);
        sketchedBitmaps.add(pencilSketchRGBBitmap);
        sketchedBitmaps.add(originalBitmap);
        sketchedBitmaps.add(originalBitmap);
        return sketchedBitmaps;
    }

    private float[][][][] bitmapToFloatArray(Bitmap bitmap) {
        int[] pixels = new int[INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_IMAGE_SIZE, 0, 0, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);

        float[][][][] arr = new float[1][INPUT_IMAGE_SIZE][INPUT_IMAGE_SIZE][3];
        for (int i=0; i<INPUT_IMAGE_SIZE; i++) {
            for(int j=0; j<INPUT_IMAGE_SIZE; j++){
                final int val = pixels[(i*INPUT_IMAGE_SIZE)+j];

                arr[0][i][j][0] = ((((val>> 16) & 0xFF)-127.5f)/127.5f);
                arr[0][i][j][1] = ((((val>> 8) & 0xFF)-127.5f)/127.5f);
                arr[0][i][j][2] = (((val & 0xFF)-127.5f)/127.5f);
            }
        }

        return arr;
    }

    private Bitmap floatArrayToBitmap(float[][][][] arr) {
        Bitmap bitmap = Bitmap.createBitmap(INPUT_IMAGE_SIZE , INPUT_IMAGE_SIZE, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE];
        for (int i=0; i<INPUT_IMAGE_SIZE; i++) {
            for(int j=0; j<INPUT_IMAGE_SIZE; j++){
                int a = 0xFF;
                float r = (arr[0][i][j][0]*127.5f)+127.5f;
                float g = (arr[0][i][j][1]*127.5f)+127.5f;
                float b = (arr[0][i][j][2]*127.5f)+127.5f;
                pixels[(i*INPUT_IMAGE_SIZE)+j] = a << 24 | (int)r << 16 | (int)g << 8 | (int)b;
            }
        }
        bitmap.setPixels(pixels, 0, INPUT_IMAGE_SIZE, 0, 0, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
        return bitmap;
    }
}