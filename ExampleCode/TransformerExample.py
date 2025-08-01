from quacknet.Transformer.TransformerManager import TransformerBlock
import numpy as np

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

inputs = np.array([[0, 1, 1, 1]])

targets = np.zeros((1, 4, embeddingDimension))
targets[0, 0, :] = 1.0
targets[0, 1, :] = 1.0
targets[0, 2, :] = 1.0
targets[0, 3, :] = 1.0

epochs = 300
learningRate = 0.01

for i in range(epochs):
    loss = block.train(inputs, targets, useBatches=False, alpha=learningRate)
    if((i + 1) % 20 == 0 or i == 0):
        print(f"Loss {i + 1}: {loss}")