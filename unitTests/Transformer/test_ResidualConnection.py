from quacknet import ResidualConnection

def test_Residual():
    output = ResidualConnection.forwardPropagation(None, 1, 2)
    assert output == 3