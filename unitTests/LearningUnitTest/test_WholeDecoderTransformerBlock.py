from quacknet import TransformerBlock
import numpy as np

def test_TransformerDecoder_CheckIfModelCanLearnOnSimpleData():
    sequenceLength = 4
    embeddingDimension = 10
    numberHeads = 2
    FFNHiddenDimension = 16
    batchSize = 1
    vocabSize = 5

    block = TransformerBlock(
        batchSize=batchSize,
        sequenceLength=sequenceLength,
        vocabSize=vocabSize,
        embedDimension=embeddingDimension,
        positionalEmbddingDimension=sequenceLength,
        numberHeads=numberHeads,
        hiddenDimensionFFN=FFNHiddenDimension,
        useResidual=True,
        useNorm=True,
        blockType="decoder"
    )
    block.firstBlock = True

    inputs = np.array([[0, 1, 2, 3]])

    targets = np.zeros((batchSize, sequenceLength, embeddingDimension))
    targets[:, :, 4] = 1.0  

    learningRate = 0.01

    initialLoss = block.train(inputs, targets, useBatches=False, alpha=learningRate)

    for _ in range(100):
        loss = block.train(inputs, targets, useBatches=False, alpha=learningRate)

    assert loss < initialLoss