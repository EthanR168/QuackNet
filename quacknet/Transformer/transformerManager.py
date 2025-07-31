from quacknet.Transformer.ResidualConnection import ResidualConnection
from quacknet.Transformer.FeedForwardNetwork import FeedForwardNetwork
from quacknet.Transformer.MultiHeadAttention import MultiAttentionHeadLayer
from quacknet.Transformer.NormLayer import NormLayer
from quacknet.Transformer.PositionalEncoding import PositionalEncoding

"""
Good website that explains transformers: https://poloclub.github.io/transformer-explainer/

Transformer layers (to add):
-   Multi head attention
-   Positional encoding 
-   Feedforward Network
-   Norm layers
-   Residual connection

Maybe will add in future:
-   Decoder
-   tokenisation (e.g., turn words into vectors)

The library will have a default transformer architecture (the one from "Attention is All You Need" paper) 
and allow the user to add any layer in any order

Default transformer architecture [called transformer block] (only the encoder section):
-   Positional Encoding
-   Multi Head Attention
-   Residual Connection
-   Layer Normalisation
-   Feed forward Network
-   Residual Connection
-   Layer Normalisation
"""

class Transformer:
    def __init__(self):
        pass

    def addLayers(self, layer): # gets a class and adds it to the layers list
        self.layers.append(layer)

    def forwardPropagation(self, inputData): # just place holder as of now
        input = inputData
        output = 0
        for layer in self.layers:
            if(layer == ResidualConnection):
                ResidualConnection.forwardPropagation(None, input, output)
                continue
            output = layer.forwardPropagation(input)

class TransformerBlock:
    def __init__(self, embedDimension, positionalEmbddingDimension, numberHeads, hiddenDimensionFFN, useResidual = True, useNorm = True):
        self.embedDimension = embedDimension
        self.numberHeads = numberHeads
        self.hiddenDimensionFFN = hiddenDimensionFFN
        self.useResidual = useResidual
        self.useNorm = useNorm
        self.positionalEmbddingDimension = positionalEmbddingDimension

        self.positionalEncoding = PositionalEncoding(
            maxDimension=positionalEmbddingDimension,
            embeddingSize=embedDimension
        )

        self.attention = MultiAttentionHeadLayer(
            embedDimension=embedDimension,
            numberOfHeads=numberHeads,
            QueryWeights=None,
            KeyWeights=None, 
            ValueWeights=None,
            outputWeight=None,
            outputBias=None,
        )

        self.FFN = FeedForwardNetwork(
            inputDimension=embedDimension,
            hiddenDimension=hiddenDimensionFFN,
            W1=None,
            b1=None,
            W2=None,
            b2=None,
        )

        if(useNorm == True):
            self.norm1 = NormLayer(embedDimension)
            self.norm2 = NormLayer(embedDimension)

    def forwardPropagation(self, input):
        input = self.positionalEncoding.forwardPropagation(input)

        attentionOutput = self.attention.forwardPropagation(input)
        if(self.useResidual == True):
            attentionOutput = ResidualConnection.forwardPropagation(None, input, attentionOutput)
        
        if(self.useNorm == True):
            attentionOutput = self.norm1.forward(attentionOutput)

        FNNOutput = self.FFN.forwardPropagation(attentionOutput)
        if(self.useResidual == True):
            FNNOutput = ResidualConnection.forwardPropagation(None, FNNOutput, attentionOutput)
        
        if(self.useNorm == True):
            FNNOutput = self.norm1.forward(FNNOutput)

        return FNNOutput
    