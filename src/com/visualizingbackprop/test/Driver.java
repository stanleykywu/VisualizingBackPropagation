package com.visualizingbackprop.test;

import tester.Tester;

public class Driver {
    public static void main(String ...args) {
        double[][] out = new double[2][1];
        double[][] in = new double[1][1];

        out[0][0] = 1;
        out[1][0] = 0;
        in[0][0] = 0.3;

        Net simple = new Net(out, false);

        for(int epoch = 0; epoch < 100; epoch++) {
            System.out.println(simple.propAndUpdate(in));
        }
    }
}
