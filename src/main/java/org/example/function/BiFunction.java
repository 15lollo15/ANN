package org.example.function;

import org.ejml.simple.SimpleMatrix;

public interface BiFunction {
    SimpleMatrix compute(SimpleMatrix x, SimpleMatrix y);
    SimpleMatrix computeDerivative(SimpleMatrix x, SimpleMatrix y);
}
