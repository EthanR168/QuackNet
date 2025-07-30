from quacknet.Transformer.ResidualConnection import ResidualConnection

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
-   Multi Head Attention
-   Residual Connection
-   Layer Normalisation
-   Feed forward Network
-   Residual Connection
-   Layer Normalisation
"""

class Transformer:
    def __init__(self, layers):
        self.layers = layers

    def forwardPropagation(self, inputData): # just place holder as of now
        input = inputData
        output = 0
        for layer in self.layers:
            if(layer == ResidualConnection):
                ResidualConnection.forwardPropagation(None, input, output)
                continue
            output = layer.forwardPropagation(input)
