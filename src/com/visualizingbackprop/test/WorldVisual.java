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

    WorldVisual(Net simpleNet) {
        this.simpleNet = simpleNet;
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

        return empty;
    }

    @Override
    public void onTick() {
        System.out.println("Loss: " + this.simpleNet.propAndUpdate());
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
