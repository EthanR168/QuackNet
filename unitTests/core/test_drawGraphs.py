from quacknet import drawGraphs
import pytest

def test_drawGraphsRaisesError():
    allAccuracy = [
    ]
    allLoss = [
    ]

    with pytest.raises(ValueError):
        drawGraphs(allAccuracy, allLoss)
    
