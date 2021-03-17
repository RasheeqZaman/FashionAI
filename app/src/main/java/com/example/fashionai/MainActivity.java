package com.example.fashionai;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private final int INPUT_IMAGE_SIZE = 128,
            PHOTO_FROM_CAMERA_REQUEST_CODE = 0,
            PHOTO_FROM_GALLERY_REQUEST_CODE = 1,
            PERMISSION_REQUEST_CODE = 2;
    ImageView imageViewOutput, imageViewInput;
    Interpreter tfLite;
    //String printData = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button btnCamera = (Button) findViewById(R.id.btnCamera);
        Button btnGallery = (Button) findViewById(R.id.btnGallery);
        imageViewInput = (ImageView) findViewById(R.id.imageViewInput);
        imageViewOutput = (ImageView) findViewById(R.id.imageViewOutput);

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

        /*if (Build.VERSION.SDK_INT >= 23) {
            int permissionCheck = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE);
            if (permissionCheck != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
            }
        }*/
        requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
    }

    private ByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
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
                Log.d("C1: ", "Enter Gallery Code");
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
                Log.d("C1: ", "Exit Gallery Code: "+pictureFile.getAbsolutePath()+": "+(inputBitmap==null));
                break;

            default:
                return;
        }

        imageViewInput.setImageBitmap(inputBitmap);
        try{
            Bitmap outputBitmap = doInterference(inputBitmap);
            imageViewOutput.setImageBitmap(outputBitmap);
            //showAlert(printData);
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
        ByteBuffer inputBuffer = bitmapToByteBuffer(inputBitmap);
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(Float.BYTES*INPUT_IMAGE_SIZE*INPUT_IMAGE_SIZE*3);
        tfLite.run(inputBuffer, outputBuffer);
        return byteBufferToBitmap(outputBuffer);
        //return byteBufferToBitmap(inputBuffer);
    }

    private ByteBuffer bitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(Float.BYTES*INPUT_IMAGE_SIZE*INPUT_IMAGE_SIZE*3);
        buffer.order(ByteOrder.nativeOrder());

        Bitmap bitmapScaled = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);
        int[] pixels = new int[INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE];
        bitmapScaled.getPixels(pixels, 0, INPUT_IMAGE_SIZE, 0, 0, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
        /*printData+=("Input pixel0: "+pixels[0]);
        printData+=(" r: "+(((pixels[0]>> 16) & 0xFF)-127.5f)/127.5f);
        printData+=(" g: "+(((pixels[0]>> 8) & 0xFF)-127.5f)/127.5f);
        printData+=(" b: "+(((pixels[0]) & 0xFF)-127.5f)/127.5f);*/

        for (int i = 0; i < INPUT_IMAGE_SIZE*INPUT_IMAGE_SIZE; i++) {
            final int val = pixels[i];

            buffer.putFloat((((val>> 16) & 0xFF)-127.5f)/127.5f);
            buffer.putFloat((((val>> 8) & 0xFF)-127.5f)/127.5f);
            buffer.putFloat(((val & 0xFF)-127.5f)/127.5f);
        }

        return buffer;
    }

    private Bitmap byteBufferToBitmap(ByteBuffer buffer) {
        /*buffer.rewind();
        printData+=(" Output r: "+buffer.getFloat());
        printData+=(" g: "+buffer.getFloat());
        printData+=(" b: "+buffer.getFloat());*/

        buffer.rewind();
        Bitmap bitmap = Bitmap.createBitmap(INPUT_IMAGE_SIZE , INPUT_IMAGE_SIZE, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE];
        for (int i = 0; i < INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE; i++) {
            int a = 0xFF;
            float r = (buffer.getFloat()*127.5f)+127.5f;
            float g = (buffer.getFloat()*127.5f)+127.5f;
            float b = (buffer.getFloat()*127.5f)+127.5f;
            pixels[i] = a << 24 | (int)r << 16 | (int)g << 8 | (int)b;
        }
        //printData+=(" Output pixel0: "+pixels[0]);
        bitmap.setPixels(pixels, 0, INPUT_IMAGE_SIZE, 0, 0, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
        return bitmap;
    }
}