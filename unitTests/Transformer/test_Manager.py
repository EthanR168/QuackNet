from quacknet import TransformerBlock
import numpy as np

def test_forward():
    sequenceLength = 4
    vocabSize = 10
    embedDimension = 8
    positionalEmbddingDimension = 8
    numberHeads = 2
    hiddenDimensionFFN = 16

    input_data = np.random.randint(low = 0, high = vocabSize, size=(1, sequenceLength))
    
    block = TransformerBlock(
        batchSize=1,
        sequenceLength=sequenceLength,
        vocabSize=vocabSize,
        embedDimension=embedDimension,
        positionalEmbddingDimension=positionalEmbddingDimension,
        numberHeads=numberHeads,
        hiddenDimensionFFN=hiddenDimensionFFN,
        useResidual=True,
        useNorm=True,
    )

    output = block.forwardPropagation(input_data)
    assert output.shape == (1, sequenceLength, embedDimension) 