from quacknet.Transformer.transformerManager import TransformerBlock
import numpy as np

sequenceLength = 4
embeddingDimension = 8
numberHeads = 2
FFNHiddenDimension = 16

inputData = np.random.rand(1, sequenceLength, embeddingDimension)

block = TransformerBlock(
    embedDimension=embeddingDimension,
    positionalEmbddingDimension=sequenceLength,
    numberHeads=numberHeads,
    hiddenDimensionFFN=FFNHiddenDimension,
    useResidual=True,
    useNorm=True
)

output = block.forwardPropagation(inputData)