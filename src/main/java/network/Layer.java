package network;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations;
import function.BiFunction;
import function.Function;
import function.impl.Sigmoid;
import function.impl.SquaredError;

import java.util.Random;

public class Layer {
    private SimpleMatrix weights;
    private SimpleMatrix biases;
    private final int numNeuron;
    private final int numInput;
    private final Function aFunction;
    private final BiFunction lossFunction;
    private static final double LEARNING_RATE = 0.01;

    public Layer(SimpleMatrix weights, SimpleMatrix biases) {
        this.numInput = weights.getNumRows();
        this.numNeuron = weights.getNumCols();
        if (biases.getNumCols() != this.numNeuron || biases.getNumRows() != 1)
            throw new RuntimeException("Biases and weights incompatible");
        this.weights = weights;
        this.biases = biases;
        this.aFunction =  new Sigmoid();
        this.lossFunction = new SquaredError();
    }

    public Layer(int numNeuron, int numInput) {
        this(numNeuron, numInput, new Sigmoid(), new SquaredError());
    }

    public Layer(int numNeuron, int numInput, Function aFunction, BiFunction lossFunction) {
        Random random = new Random(15);
        this.numNeuron = numNeuron;
        this.numInput = numInput;
        this.aFunction = aFunction;
        this.lossFunction = lossFunction;
        double bound = 1 / Math.sqrt(numInput);
        weights = new SimpleMatrix(numInput, numNeuron);
        weights = weights.elementOp((SimpleOperations.ElementOpReal) (i, j, v) -> random.nextDouble(-bound, bound));

        biases = new SimpleMatrix(1, numNeuron);
    }

    public SimpleMatrix predict(SimpleMatrix x) {
        return aFunction.compute(x.mult(weights).plus(biases));
    }

    public SimpleMatrix train(SimpleMatrix x, SimpleMatrix y) {
        SimpleMatrix sum = x.mult(weights).plus(biases);
        SimpleMatrix yPredicted = predict(x);

        SimpleMatrix dLoss = lossFunction.computeDerivative(yPredicted, y);
        SimpleMatrix dActivation = aFunction.computeDerivative(sum);

        SimpleMatrix backup = dLoss.elementMult(dActivation).mult(weights.transpose());

        SimpleMatrix wDirection = dLoss.elementMult(dActivation).transpose().mult(x).scale(LEARNING_RATE);
        SimpleMatrix bDirection = dLoss.elementMult(dActivation).scale(LEARNING_RATE);

        weights = weights.minus(wDirection.transpose());
        biases = biases.minus(bDirection);


        return backup;
    }

    public SimpleMatrix trainHidden(SimpleMatrix x, SimpleMatrix prevDerivative) {
        SimpleMatrix sum = x.mult(weights).plus(biases);
        SimpleMatrix dActivation = aFunction.computeDerivative(sum);

        SimpleMatrix backup = prevDerivative.elementMult(dActivation).mult(weights.transpose());

        SimpleMatrix wDirection = prevDerivative.elementMult(dActivation).transpose().mult(x).scale(LEARNING_RATE);
        SimpleMatrix bDirection = prevDerivative.elementMult(dActivation).scale(LEARNING_RATE);
        weights = weights.minus(wDirection.transpose());
        biases = biases.minus(bDirection);


        return backup;
    }

    public int getNumNeuron() {
        return numNeuron;
    }

    public int getNumInput() {
        return numInput;
    }

    public SimpleMatrix getWeights() {
        return weights;
    }


    public SimpleMatrix getBiases() {
        return biases;
    }

}
