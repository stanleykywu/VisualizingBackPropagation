package com.visualizingbackprop.test;

import javalib.impworld.World;
import javalib.impworld.WorldScene;
import javalib.worldimages.*;

import java.awt.*;

interface IConstants {
    WorldImage node = new CircleImage(20, OutlineMode.SOLID, Color.darkGray);
    WorldImage background = new RectangleImage(500, 500, OutlineMode.SOLID, Color.WHITE);
}

public class WorldVisual extends World {
    Net simpleNet;
    double loss;

    WorldVisual(Net simpleNet) {
        this.simpleNet = simpleNet;
        this.loss = 0;
    }

    @Override
    public WorldScene makeScene() {
        WorldScene empty = this.getEmptyScene();
        empty.placeImageXY(IConstants.background, 250, 250);

        empty.placeImageXY(new RectangleImage(175, 4, OutlineMode.SOLID,
                this.scaleColor(this.simpleNet.inputWeights[0][0])), 162, 150);
        empty.placeImageXY(new RectangleImage(175, 4, OutlineMode.SOLID,
                this.scaleColor(this.simpleNet.inputWeights[1][1])), 162, 350);
        empty.placeImageXY(new RectangleImage(175, 4, OutlineMode.SOLID,
                this.scaleColor(this.simpleNet.hiddenWeights[0][0])), 337, 150);
        empty.placeImageXY(new RectangleImage(175, 4, OutlineMode.SOLID,
                this.scaleColor(this.simpleNet.hiddenWeights[1][1])), 337, 350);

        empty.placeImageXY(new RotateImage(new RectangleImage(265, 4, OutlineMode.SOLID,
                this.scaleColor(this.simpleNet.inputWeights[1][0])), 48), 162, 250);
        empty.placeImageXY(new RotateImage(new RectangleImage(265, 4, OutlineMode.SOLID,
                this.scaleColor(this.simpleNet.hiddenWeights[1][0])), 48), 337, 250);
        empty.placeImageXY(new RotateImage(new RectangleImage(265, 4, OutlineMode.SOLID,
                this.scaleColor(this.simpleNet.inputWeights[0][1])), 132), 162, 250);
        empty.placeImageXY(new RotateImage(new RectangleImage(265, 4, OutlineMode.SOLID,
                this.scaleColor(this.simpleNet.hiddenWeights[0][1])), 132), 337, 250);

        empty.placeImageXY(IConstants.node, 75, 150);
        empty.placeImageXY(IConstants.node, 75, 350);
        empty.placeImageXY(IConstants.node, 250, 150);
        empty.placeImageXY(IConstants.node, 250, 350);
        empty.placeImageXY(IConstants.node, 425, 150);
        empty.placeImageXY(IConstants.node, 425, 350);

        empty.placeImageXY(this.displayInputWeights(), 166, 400);
        empty.placeImageXY(this.displayHiddenWeights(), 333, 400);
        empty.placeImageXY(this.displayLoss(), 150, 475);
        empty.placeImageXY(this.displayAccuracy(), 350, 475);
        return empty;
    }

    public WorldImage displayInputWeights() {
        WorldImage inputWeightRow0 = new TextImage(String.format("%.3g%n",
                this.simpleNet.inputWeights[0][0]) + "   " + String.format("%.3g%n", this.simpleNet.inputWeights[0][1]), 20, Color.BLACK);
        WorldImage inputWeightRow1 = new TextImage(String.format("%.3g%n",
                this.simpleNet.inputWeights[1][0]) + "   " + String.format("%.3g%n", this.simpleNet.inputWeights[1][1]), 20, Color.BLACK);

        WorldImage inputWeightResult = new AboveImage(inputWeightRow0, inputWeightRow1);

        return inputWeightResult;
    }

    public WorldImage displayHiddenWeights() {
        WorldImage hiddenWeightRow0 = new TextImage(String.format("%.3g%n",
                this.simpleNet.hiddenWeights[0][0]) + "   " + String.format("%.3g%n", this.simpleNet.hiddenWeights[0][1]), 20, Color.BLACK);
        WorldImage hiddenWeightRow1 = new TextImage(String.format("%.3g%n",
                this.simpleNet.hiddenWeights[1][0]) + "   " + String.format("%.3g%n", this.simpleNet.hiddenWeights[1][1]), 20, Color.BLACK);

        WorldImage hiddenWeightResult = new AboveImage(hiddenWeightRow0, hiddenWeightRow1);

        return hiddenWeightResult;
    }

    public WorldImage displayLoss() {
        WorldImage lossText = new TextImage("Loss: " + String.format("%.3g%n", this.loss), 25, Color.black);

        return lossText;
    }

    public WorldImage displayAccuracy() {
        WorldImage accuracyText = new TextImage("Accuracy: " + String.format("%.3g%n", this.simpleNet.computeAccuracy()), 25, Color.black);

        return accuracyText;
    }

    @Override
    public void onTick() {
        this.loss =this.simpleNet.stochasticBackPropagation();
    }

    public Color scaleColor(double weight) {
        int whiteScale = 255 - Math.min(Math.max((int) (weight  * 255), 25), 255);
        return new Color(whiteScale, whiteScale, whiteScale);
    }

    @Override
    public WorldEnd worldEnds() {
        return super.worldEnds();
    }
}
