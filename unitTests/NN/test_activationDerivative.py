from  neuralLibrary.activationDerivativeFunctions import ReLUDerivative, SigmoidDerivative, TanHDerivative, LinearDerivative, SoftMaxDerivative
import numpy as np

def test_ReLUDerivative():
    assert np.allclose(ReLUDerivative(np.array([1, 1])), np.array([1, 1]))
    assert np.allclose(ReLUDerivative(np.array([-1, 0])), np.array([0.01, 0.01]))

def test_SigmoidDerivative():
    assert np.allclose(SigmoidDerivative(np.array([1, -1])), np.array([0.196611933241, 0.196611933241]))
    assert np.allclose(SigmoidDerivative(np.array([0, 0])), np.array([0.25, 0.25]))

def test_TanHDerivative():
    assert np.allclose(TanHDerivative(np.array([0, 1, -1])), np.array([1, 0.419974341614, 0.419974341614]))
    assert np.allclose(TanHDerivative(np.array([1000, 2, -0.001])), np.array([0, 0.0706508248532, 0.999999000001]))

def test_LinearDerivative():
    assert np.allclose(LinearDerivative(np.array([-10000000, 0, 0.00000001, 47747824787374647826347624786237486823, np.pi])), np.array([1, 1, 1, 1, 1]))

def test_SoftMaxDerivativeWithCrossEntropy():
    assert np.allclose(SoftMaxDerivative(np.array([0.25, -1, 0.6]), np.array([1, 0, -1])), np.array([0.75, 1, -1.6]))