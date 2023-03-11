package function;

import org.ejml.simple.SimpleMatrix;

public interface Function {
    SimpleMatrix compute(SimpleMatrix x);
    SimpleMatrix computeDerivative(SimpleMatrix x);
}
