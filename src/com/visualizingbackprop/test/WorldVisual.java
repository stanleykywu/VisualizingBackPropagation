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

        empty.placeImageXY(new LineImage(new Posn(175, -100),
                this.scaleColor(this.simpleNet.inputWeights[0][0])), 138, 200);
        empty.placeImageXY(new LineImage(new Posn(175, 100),
                this.scaleColor(this.simpleNet.inputWeights[1][0])), 138, 300);
        empty.placeImageXY(new LineImage(new Posn(175, 0),
                this.scaleColor(this.simpleNet.hiddenWeights[0][0])), 312, 150);
        empty.placeImageXY(new LineImage(new Posn(175, 200),
                this.scaleColor(this.simpleNet.hiddenWeights[1][0])), 312, 250);
        empty.placeImageXY(new LineImage(new Posn(175, -200),
                this.scaleColor(this.simpleNet.hiddenWeights[0][1])), 312, 250);
        empty.placeImageXY(new LineImage(new Posn(175, 0),
                this.scaleColor(this.simpleNet.hiddenWeights[1][1])), 312, 350);

        empty.placeImageXY(IConstants.node, 50, 250);
        empty.placeImageXY(IConstants.node, 225, 150);
        empty.placeImageXY(IConstants.node, 225, 350);
        empty.placeImageXY(IConstants.node, 400, 150);
        empty.placeImageXY(IConstants.node, 400, 350);

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
