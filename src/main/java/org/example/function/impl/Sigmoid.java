package org.example.function.impl;

import org.ejml.simple.SimpleMatrix;
import org.example.function.Function;

public class Sigmoid implements Function {
    @Override
    public SimpleMatrix compute(SimpleMatrix x) {
        return x.elementOp((i, j, v) -> 1 / (1 + Math.exp(-v)));
    }

    @Override
    public SimpleMatrix computeDerivative(SimpleMatrix x) {
        SimpleMatrix tmp = compute(x);
        return tmp.elementOp((i, j, v) -> v * (1 - v));
    }

}
