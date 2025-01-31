package com.example.yolov8_detect;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class RectView extends View {
    private final Map<RectF, String> negeri = new HashMap<>();
    private final Map<RectF, String> kampung = new HashMap<>();
    private final Map<RectF, String> bebek = new HashMap<>(); // Peta untuk objek ketiga

    private final Paint firePaint = new Paint();
    private final Paint smokePaint = new Paint();
    private final Paint thirdPaint = new Paint(); // Paint untuk objek ketiga
    private final Paint textPaint = new Paint();

    private String[] labels;

    public RectView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);

        firePaint.setStyle(Paint.Style.STROKE);
        firePaint.setStrokeWidth(10.0f);
        firePaint.setColor(Color.RED);
        firePaint.setStrokeCap(Paint.Cap.ROUND);
        firePaint.setStrokeJoin(Paint.Join.ROUND);
        firePaint.setStrokeMiter(100);

        smokePaint.setStyle(Paint.Style.STROKE);
        smokePaint.setStrokeWidth(10.0f);
        smokePaint.setColor(Color.GREEN);
        smokePaint.setStrokeCap(Paint.Cap.ROUND);
        smokePaint.setStrokeJoin(Paint.Join.ROUND);
        smokePaint.setStrokeMiter(100);

        thirdPaint.setStyle(Paint.Style.STROKE);
        thirdPaint.setStrokeWidth(10.0f);
        thirdPaint.setColor(Color.BLUE); // Warna untuk objek ketiga
        thirdPaint.setStrokeCap(Paint.Cap.ROUND);
        thirdPaint.setStrokeJoin(Paint.Join.ROUND);
        thirdPaint.setStrokeMiter(100);

        textPaint.setTextSize(60.0f);
        textPaint.setColor(Color.WHITE);
    }

    public void setLabels(String[] labels) {
        this.labels = labels;
    }

    public ArrayList<Result> transFormRect(ArrayList<Result> resultArrayList) {
        float scaleX = getWidth() / (float) SupportOnnx.INPUT_SIZE;
        float scaleY = scaleX * 9f / 16f;
        float realY = getWidth() * 9f / 16f;
        float diffY = realY - getHeight();

        for (Result result : resultArrayList) {
            result.getRectF().left *= scaleX;
            result.getRectF().right *= scaleX;
            result.getRectF().top = result.getRectF().top * scaleY - (diffY / 2f);
            result.getRectF().bottom = result.getRectF().bottom * scaleY - (diffY / 2f);
        }
        return resultArrayList;
    }

    public void clear() {
        negeri.clear();
        kampung.clear();
        bebek.clear(); // Bersihkan peta untuk objek ketiga
    }

    public void resultToList(ArrayList<Result> results) {
        for (Result result : results) {
            if (result.getLabel() == 0) { // fire
                negeri.put(result.getRectF(), labels[0] + ", " + Math.round(result.getScore() * 100) + "%");
            } else if (result.getLabel() == 1) { // smoke
                kampung.put(result.getRectF(), labels[1] + ", " + Math.round(result.getScore() * 100) + "%");
            } else if (result.getLabel() == 2) { // third object
                bebek.put(result.getRectF(), labels[2] + ", " + Math.round(result.getScore() * 100) + "%");
            }
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        for (Map.Entry<RectF, String> fire : negeri.entrySet()) {
            canvas.drawRect(fire.getKey(), firePaint);
            canvas.drawText(fire.getValue(), fire.getKey().left + 10.0f, fire.getKey().top + 60.0f, textPaint);
        }
        for (Map.Entry<RectF, String> smoke : kampung.entrySet()) {
            canvas.drawRect(smoke.getKey(), smokePaint);
            canvas.drawText(smoke.getValue(), smoke.getKey().left + 10.0f, smoke.getKey().top + 60.0f, textPaint);
        }
        for (Map.Entry<RectF, String> third : bebek.entrySet()) {
            canvas.drawRect(third.getKey(), thirdPaint);
            canvas.drawText(third.getValue(), third.getKey().left + 10.0f, third.getKey().top + 60.0f, textPaint);
        }
        super.onDraw(canvas);
    }
}
