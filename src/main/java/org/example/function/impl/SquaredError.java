package org.example.function.impl;

import org.ejml.simple.SimpleMatrix;
import org.example.function.BiFunction;

public class SquaredError  implements BiFunction {
    @Override
    public SimpleMatrix compute(SimpleMatrix x, SimpleMatrix y) {
        return y.minus(x).elementPower(2);
    }

    @Override
    public SimpleMatrix computeDerivative(SimpleMatrix x, SimpleMatrix y) {
        return y.minus(x).scale(-2);
    }

}
