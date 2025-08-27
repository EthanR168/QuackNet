from quacknet import TransformerBlock
import numpy as np

def test_Transformer_CheckIfModelCanLearnOnSimpleData():
    sequenceLength = 4
    embeddingDimension = 10
    numberHeads = 2
    FFNHiddenDimension = 16
    batchSize = 1
    vocabSize = 2

    block = TransformerBlock(
        batchSize=batchSize,
        sequenceLength=sequenceLength,
        vocabSize=vocabSize,
        embedDimension=embeddingDimension,
        positionalEmbddingDimension=sequenceLength,
        numberHeads=numberHeads,
        hiddenDimensionFFN=FFNHiddenDimension,
        useResidual=True,
        useNorm=True
    )
    block.firstBlock = True

    inputs = np.array([[0, 1, 1, 1]])

    targets = np.zeros((1, 4, embeddingDimension))
    targets[0, 0, :] = 1.0
    targets[0, 1, :] = 1.0
    targets[0, 2, :] = 1.0
    targets[0, 3, :] = 1.0

    learningRate = 0.01

    initialLoss = block.train(inputs, targets, useBatches=False, alpha=learningRate)

    for i in range(100):
        loss = block.train(inputs, targets, useBatches=False, alpha=learningRate)
    
    assert loss < initialLoss
