// NEW VERSION

package org.firstinspires.ftc.teamcode;

import com.acmerobotics.dashboard.FtcDashboard;
import com.acmerobotics.dashboard.telemetry.MultipleTelemetry;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.openftc.easyopencv.OpenCvCamera;
import org.openftc.easyopencv.OpenCvCameraFactory;
import org.openftc.easyopencv.OpenCvCameraRotation;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;

@TeleOp(name = "OpenCV Testing")
public class opencv extends LinearOpMode {

    private OpenCvCamera controlHubCam;
    private static final int CAMERA_WIDTH = 640;
    private static final int CAMERA_HEIGHT = 360;

    // Real-world object width and camera focal length for distance calculation
    public static final double objectWidthInRealWorldUnits = 3.75;  // inches
    public static final double focalLength = 728;  // pixels

    // Pipeline instance to access detection data
    private ColorBlobDetectionPipeline pipeline;

    @Override
    public void runOpMode() {
        initOpenCV();
        FtcDashboard dashboard = FtcDashboard.getInstance();
        telemetry = new MultipleTelemetry(telemetry, dashboard.getTelemetry());
        FtcDashboard.getInstance().startCameraStream(controlHubCam, 30);

        waitForStart();

        while (opModeIsActive()) {
            String detectedColors = "";
            if(pipeline.detectedYellow) {
                detectedColors += "Yellow (" + (int)pipeline.yellowCX + ", " + (int)pipeline.yellowCY + ") ";
            }
            if(pipeline.detectedBlue) {
                detectedColors += "Blue (" + (int)pipeline.blueCX + ", " + (int)pipeline.blueCY + ") ";
            }
            if(pipeline.detectedRed) {
                detectedColors += "Red (" + (int)pipeline.redCX + ", " + (int)pipeline.redCY + ") ";
            }
            if(detectedColors.isEmpty()) detectedColors = "None";

            telemetry.addData("Detected Colors", detectedColors);
            telemetry.update();
        }

        controlHubCam.stopStreaming();
    }

    private void initOpenCV() {
        int cameraMonitorViewId = hardwareMap.appContext.getResources().getIdentifier(
                "cameraMonitorViewId", "id", hardwareMap.appContext.getPackageName());

        controlHubCam = OpenCvCameraFactory.getInstance().createWebcam(
                hardwareMap.get(WebcamName.class, "Webcam 1"), cameraMonitorViewId);

        pipeline = new ColorBlobDetectionPipeline();
        controlHubCam.setPipeline(pipeline);

        controlHubCam.openCameraDevice();
        controlHubCam.startStreaming(CAMERA_WIDTH, CAMERA_HEIGHT, OpenCvCameraRotation.UPRIGHT);
    }

    // Pipeline class for color blob detection
    class ColorBlobDetectionPipeline extends OpenCvPipeline {

        // Per color detected info:
        public double yellowCX = -1, yellowCY = -1, yellowWidth = 0;
        public double blueCX = -1, blueCY = -1, blueWidth = 0;
        public double redCX = -1, redCY = -1, redWidth = 0;

        public boolean detectedYellow = false;
        public boolean detectedBlue = false;
        public boolean detectedRed = false;

        // Temporary vars to pass info back from processColorBlob
        private double tempCX, tempCY, tempWidth;

        @Override
        public Mat processFrame(Mat input) {
            Mat hsvFrame = new Mat();
            Imgproc.cvtColor(input, hsvFrame, Imgproc.COLOR_BGR2HSV);

            // Yellow detection
            Scalar lowerYellow = new Scalar(197, 255, 255);
            Scalar upperYellow = new Scalar(0, 128, 139);
            Mat yellowMask = new Mat();
            Core.inRange(hsvFrame, lowerYellow, upperYellow, yellowMask);
            detectedYellow = processColorBlob(yellowMask, input, new Scalar(0, 255, 255));
            if (detectedYellow) {
                yellowCX = tempCX;
                yellowCY = tempCY;
                yellowWidth = tempWidth;
            } else {
                yellowCX = yellowCY = -1;
                yellowWidth = 0;
            }

            // Blue detection
            Scalar lowerBlue = new Scalar(100, 150, 0);
            Scalar upperBlue = new Scalar(140, 255, 255);
            Mat blueMask = new Mat();
            Core.inRange(hsvFrame, lowerBlue, upperBlue, blueMask);
            detectedBlue = processColorBlob(blueMask, input, new Scalar(255, 0, 0));
            if (detectedBlue) {
                blueCX = tempCX;
                blueCY = tempCY;
                blueWidth = tempWidth;
            } else {
                blueCX = blueCY = -1;
                blueWidth = 0;
            }

            // Red detection (two HSV ranges merged)
            Scalar lowerRed1 = new Scalar(0, 120, 70);
            Scalar upperRed1 = new Scalar(10, 255, 255);
            Scalar lowerRed2 = new Scalar(170, 120, 70);
            Scalar upperRed2 = new Scalar(180, 255, 255);
            Mat redMask1 = new Mat();
            Mat redMask2 = new Mat();
            Core.inRange(hsvFrame, lowerRed1, upperRed1, redMask1);
            Core.inRange(hsvFrame, lowerRed2, upperRed2, redMask2);
            Mat redMask = new Mat();
            Core.addWeighted(redMask1, 1.0, redMask2, 1.0, 0.0, redMask);
            detectedRed = processColorBlob(redMask, input, new Scalar(0, 0, 255));
            if (detectedRed) {
                redCX = tempCX;
                redCY = tempCY;
                redWidth = tempWidth;
            } else {
                redCX = redCY = -1;
                redWidth = 0;
            }

            return input;
        }

        private boolean processColorBlob(Mat mask, Mat input, Scalar color) {
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            MatOfPoint largestContour = null;
            double maxArea = 0;
            double minArea = 500; // Adjust as needed

            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);
                if (area > minArea && area > maxArea) {
                    maxArea = area;
                    largestContour = contour;
                }
            }

            if (largestContour != null) {
                Imgproc.drawContours(input, contours, contours.indexOf(largestContour), color, 2);

                Rect boundingRect = Imgproc.boundingRect(largestContour);
                double width = boundingRect.width;

                Moments moments = Imgproc.moments(largestContour);
                double cX = moments.get_m10() / moments.get_m00();
                double cY = moments.get_m01() / moments.get_m00();

                String label = "(" + (int)cX + ", " + (int)cY + ")";
                Imgproc.putText(input, label, new Point(cX + 10, cY), Imgproc.FONT_HERSHEY_COMPLEX, 0.5, color, 2);

                String widthLabel = "Width: " + (int) width + " pixels";
                Imgproc.putText(input, widthLabel, new Point(cX + 10, cY + 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

                String distanceLabel = "Distance: " + String.format("%.2f", getDistance(width)) + " inches";
                Imgproc.putText(input, distanceLabel, new Point(cX + 10, cY + 60), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

                Imgproc.circle(input, new Point(cX, cY), 5, color, -1);

                // Save detected info to temp vars for reading in processFrame
                tempCX = cX;
                tempCY = cY;
                tempWidth = width;

                return true;
            }

            return false;
        }
    }

    private static double getDistance(double width){
        return (objectWidthInRealWorldUnits * focalLength) / width;
    }
}
