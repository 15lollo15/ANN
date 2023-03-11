package network;

import function.impl.Sigmoid;
import function.impl.SquaredError;

public class Main {
    public static void main(String[] args) {
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(
                new Layer(2, 2, new Sigmoid(), new SquaredError())
        );
        mlp.addLayer(new Layer(1, 2, new Sigmoid(), new SquaredError()));
    }
}
