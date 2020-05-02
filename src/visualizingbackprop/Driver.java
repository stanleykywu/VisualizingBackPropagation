package visualizingbackprop;

import java.util.ArrayList;
import java.util.Random;

public class Driver {

  public static void main(String... args) {
    ArrayList<double[][]> inputs = generateRandomPoints(2500);
    ArrayList<double[][]> expOut = generateCorrespondingOutput(inputs);

    Net simple = new Net(inputs, expOut, true);

    WorldVisual simpleVisual = new WorldVisual(simple);
    simpleVisual.bigBang(500, 600, 0.1);
  }

  public static ArrayList<double[][]> generateRandomPoints(int numPoints) {
    ArrayList<double[][]> result = new ArrayList<double[][]>();
    for (int i = 0; i < numPoints; i++) {
      double[][] point = new double[2][1];
      int xRange = 100;
      int yRange = 100;

      point[0][0] = (double) (new Random().nextInt(xRange) - 50) / 50;
      point[1][0] = (double) (new Random().nextInt(yRange) - 50) / 50;

      result.add(point);
    }
    return result;
  }

  public static ArrayList<double[][]> generateCorrespondingOutput(ArrayList<double[][]> input) {
    ArrayList<double[][]> result = new ArrayList<double[][]>();

    for (double[][] point : input) {
      double[][] resultPoint = new double[2][1];
      double x = point[0][0] * 100;
      double y = point[1][0] * 100;
      if (y > Math.pow(x, 2)) {
        resultPoint[0][0] = 1;
      } else {
        resultPoint[0][0] = 0.5;
      }
      result.add(resultPoint);
    }
    return result;
  }
}