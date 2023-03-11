package function.impl;

import org.ejml.simple.SimpleMatrix;
import function.Function;

public class ReLU implements Function {
    @Override
    public SimpleMatrix compute(SimpleMatrix x) {
        return x.elementOp((i, j, v) -> v >= 0 ? v : 0);
    }

    @Override
    public SimpleMatrix computeDerivative(SimpleMatrix x) {
        return x.elementOp((i, j, v) -> v >= 0 ? 1 : 0);
    }

}
