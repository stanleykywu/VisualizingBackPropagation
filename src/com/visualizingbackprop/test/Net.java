package com.visualizingbackprop.test;
import tester.Tester;

import java.util.Random;
import java.util.function.Function;

class Utility {
    double[][] multiplyMatrices(double[][] firstMatrix, double[][] secondMatrix) {
        double[][] result = new double[firstMatrix.length][secondMatrix[0].length];

        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[row].length; col++) {
                result[row][col] = multiplyMatricesCell(firstMatrix, secondMatrix, row, col);
            }
        }

        return result;
    }

    double multiplyMatricesCell(double[][] firstMatrix, double[][] secondMatrix, int row, int col) {
        double cell = 0;
        for (int i = 0; i < secondMatrix.length; i++) {
            cell += firstMatrix[row][i] * secondMatrix[i][col];
        }
        return cell;
    }

    double[][] map(double[][] input, Function<Double, Double> func) {
        double[][] result = new double[input.length][input[0].length];
        for(int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                result[i][j] = func.apply(input[i][j]);
            }
        }
        return result;
    }

    double loss(double output, double exp) {
        return Math.pow(output - exp, 2);
    }
}

class Net {
    double[][] inputWeights;
    double[][] hiddenWeights;
    double[][] expectedOut;
    boolean activation;

    Net(double[][] expectedOut, boolean activ) {
        this.expectedOut = expectedOut;
        this.activation = activ;
        this.inputWeights = new double[2][1];
        this.hiddenWeights = new double[2][2];

        for (int i = 0; i < inputWeights.length; i++) {
            for(int j = 0; j < inputWeights[0].length; j++) {
                inputWeights[i][j] = new Random().nextFloat();
            }
        }
        for (int i = 0; i < hiddenWeights.length; i++) {
            for(int j = 0; j < hiddenWeights[0].length; j++) {
                hiddenWeights[i][j] = new Random().nextFloat();
            }
        }
    }

    public double sigmoid(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }

    public double rectLinear(double x) {
        if (x <= 0) {
            return 0;
        } else {
            return x;
        }
    }

    public double sigmoidDeriv(double x) {
        return this.sigmoid(x) * (1 - this.sigmoid(x));
    }

    public double rectLinearDeriv(double x) {
        if (x <= 0) {
            return 0;
        }
        else {
            return 1;
        }
    }

    public double propAndUpdate(double[][] inputs) {
        if (this.activation) {
            return this.propAndUpdateFunc(inputs, (x -> this.sigmoid(x)), (x -> this.sigmoidDeriv(x)));
        } else {
            return this.propAndUpdateFunc(inputs, (x -> this.rectLinear(x)), (x -> this.rectLinearDeriv(x)));
        }
    }

    public double propAndUpdateFunc(double[][] inputs, Function<Double, Double> activation,
                                       Function<Double, Double> activationDer) {
        Utility utils = new Utility();

        double[][] inA = utils.multiplyMatrices(this.inputWeights, inputs);
        double[][] outA = utils.map(inA, activation);

        double[][] inB = utils.multiplyMatrices(this.hiddenWeights, outA);
        double[][] outB = utils.map(inB, activation);

        double totalLossOne = utils.loss(outB[0][0], expectedOut[0][0]);
        double totalLossTwo = utils.loss(outB[1][0], expectedOut[1][0]);

        double deltaOut00 = 2 * (outB[0][0] - expectedOut[0][0]) * activationDer.apply(inB[0][0]) * outA[0][0];
        double deltaOut01 = 2 * (outB[0][0] - expectedOut[0][0]) * activationDer.apply(inB[1][0]) * outA[0][0];
        double deltaOut10 = 2 * (outB[1][0] - expectedOut[1][0]) * activationDer.apply(inB[0][0]) * outA[1][0];
        double deltaOut11 = 2 * (outB[1][0] - expectedOut[1][0]) * activationDer.apply(inB[1][0]) * outA[1][0];

        double deltaA1 = (deltaOut00 + deltaOut10) * activationDer.apply(inA[0][0]) * inputs[0][0];
        double deltaA2 = (deltaOut01 + deltaOut11) * activationDer.apply(inA[1][0]) * inputs[0][0];

        this.inputWeights[0][0] -= deltaA1;
        this.inputWeights[1][0] -= deltaA2;

        this.hiddenWeights[0][0] -= deltaOut00;
        this.hiddenWeights[0][1] -= deltaOut01;
        this.hiddenWeights[1][0] -= deltaOut10;
        this.hiddenWeights[1][1] -= deltaOut11;

        return totalLossOne + totalLossTwo;
    }
}

