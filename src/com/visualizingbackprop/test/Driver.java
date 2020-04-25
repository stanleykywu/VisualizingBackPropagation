package com.visualizingbackprop.test;

import java.util.ArrayList;
import java.util.Random;

public class Driver {
    public static void main(String ...args) {
        ArrayList<double[][]> inputs = generateRandomPoints(100);
        ArrayList<double[][]> expOut = generateCorrespondingOutput(inputs);

        Net simple = new Net(inputs, expOut, false);

        WorldVisual simpleVisual = new WorldVisual(simple);
        simpleVisual.bigBang(500, 600, 0.2);
    }

    public static ArrayList<double[][]> generateRandomPoints(int numPoints) {
        ArrayList<double[][]> result = new ArrayList<double[][]>();
        for (int i = 0; i < numPoints; i++) {
            double[][] point = new double[2][1];
            int xRange = 200;
            int yRange = 200;

            point[0][0] = (double) (new Random().nextInt(xRange) - 100) / 100;
            point[1][0] = (double) (new Random().nextInt(yRange) - 100) / 100;

            result.add(point);
        }
        return result;
    }

    public static ArrayList<double[][]> generateCorrespondingOutput(ArrayList<double[][]> input) {
        ArrayList<double[][]> result = new ArrayList<double[][]>();

        for(double[][] point : input) {
            double[][] resultPoint = new double[2][1];
            double x = point[0][0] * 100;
            double y = point[1][0] * 100;
            if (Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2)) <= 50) {
                resultPoint[0][0] = 1;
                resultPoint[1][0] = 0;
            } else {
                resultPoint[0][0] = 0;
                resultPoint[1][0] = 1;
            }
            result.add(resultPoint);
        }
        return result;
    }
}
