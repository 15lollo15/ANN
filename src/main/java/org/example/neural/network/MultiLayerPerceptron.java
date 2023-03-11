package org.example.neural.network;

import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;

public class MultiLayerPerceptron {
    private List<Layer> layers;
    private int numInput;

    public MultiLayerPerceptron(Layer inputLayer) {
        this.numInput = inputLayer.getNumInput();
        this.layers = new ArrayList<>();
        layers.add(inputLayer);
    }

    private Layer getOutputLayer() {
        return layers.get(layers.size() - 1);
    }

    public void addLayer(Layer layer) {
        Layer lastAddedLayer = getOutputLayer();

        if (layer.getNumInput() != lastAddedLayer.getNeuronNumber())
            throw new RuntimeException("...");

        layers.add(layer);
    }

    public int getNumOutput() {
        return getOutputLayer().getNeuronNumber();
    }

    private SimpleMatrix predict(SimpleMatrix x, List<SimpleMatrix> d) {
        if (d == null)
            d = new ArrayList<>();
        for (Layer layer : layers) {
            d.add(x);
            x = layer.predict(x);
        }
        return x;
    }

    public SimpleMatrix predict(SimpleMatrix x) {
        return predict(x, null);
    }

    public void train(SimpleMatrix x, SimpleMatrix y) {
        List<SimpleMatrix> outputs = new ArrayList<>();
        predict(x, outputs);

        Layer outputLayer = getOutputLayer();
        SimpleMatrix prevD = outputLayer.train(outputs.get(outputs.size() - 1), y);

        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer layer = layers.get(i);
            SimpleMatrix in = outputs.get(i);
            prevD = layer.trainHidden(in, prevD);
        }

    }

    public int getNumInput() {
        return numInput;
    }
}
