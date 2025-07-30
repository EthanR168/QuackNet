import numpy as np

"""
Q = X @ W_Q

W_Q = Weights for query


K = X @ W_K

W_K = Weights for Key


V = X @ W_V

W_V = Weights for Value


a(Q, K, V) = softmax( (Q @ K.T) / sqrt(d) ) @ V 

a = attention
d = dimension of the key vector


O = at @ W_O + W_B

O = output
at = combined attention
W_O = output projection weight
W_O = output projection bias

Q = Query
K = Key
V = Value
X = Input embedding
@ = matrix multiplication

Order of forward prop:
-   QKV Linear projection (get QKV)
-   Split into heads (divide QKV)
-   Compute attention per Head
-   Combine all the attention 
-   Output linear projection (basically a dense layer but for 3D tensor)
"""

class MultiAttentionHeadLayer:
    def __init__(self, embedDimension, numberOfHeads, QueryWeights, KeyWeights, ValueWeights, outputWeight, outputBias):
        self.embedDimension = embedDimension
        self.numberOfHeads = numberOfHeads
        self.QueryWeights = QueryWeights
        self.KeyWeights = KeyWeights
        self.ValueWeights = ValueWeights
        self.outputWeight = outputWeight
        self.outputBias = outputBias

    def QKVLinearProjection(self, inputEmbedding):
        Query = inputEmbedding @ self.QueryWeights
        Key = inputEmbedding @ self.KeyWeights
        Value = inputEmbedding @ self.ValueWeights
        return Query, Key, Value
    
    def SplitIntoHeads(self, Query, Key, Value):
        # QVK has a shape of (batchSize, sequenceLength, embedDimension)
        headDimension = self.embedDimension // self.numberOfHeads # // returns a whole number (floor division)
        batchSize = Query.shape[0]
        sequenceLength = Query.shape[1]

        # reshape to split heads
        QReshaped = Query.reshape(batchSize, sequenceLength, self.numberOfHeads, headDimension)
        KReshaped = Key.reshape(batchSize, sequenceLength, self.numberOfHeads, headDimension)
        VReshaped = Value.reshape(batchSize, sequenceLength, self.numberOfHeads, headDimension)
        
        # Transpose to (batchSize, numberHeads, sequenceLength, headDimension)
        QHead = QReshaped.transpose(0, 2, 1, 3)
        KHead = KReshaped.transpose(0, 2, 1, 3)
        VHead = VReshaped.transpose(0, 2, 1, 3)

        return QHead, KHead, VHead
    
    def _TransformerSoftMax(self, values): # softmax in quacknet.core works for 1D arrays not 3D/4D tensors
        values = np.array(values, dtype=np.float64)
        maxVal = np.max(values, axis=-1, keepdims=True)
        values = values - maxVal
        summ = np.sum(np.exp(values), axis=-1, keepdims=True)
        out = np.exp(values) / summ
        return out

    def _calculateAttentionForOneHead(self, QueryHead, KeyHead, ValueHead):
        # a(Q, K, V) = softmax( (Q @ K.T) / sqrt(d) ) @ V 
        attentionScore = (QueryHead @ KeyHead.transpose(0, 2, 1)) / np.sqrt(ValueHead.shape[1])
        attentionOutput = self._TransformerSoftMax(attentionScore) @ ValueHead
        return attentionOutput

    def calculateAttention(self, QueryHead, KeyHead, ValueHead):
        attentionHeads = []
        for i in range(self.numberOfHeads):
            att = self._calculateAttentionForOneHead(QueryHead[:, i, :, :], KeyHead[:, i, :, :], ValueHead[:, i, :, :])
            attentionHeads.append(att)
        stackedHeads = np.stack(attentionHeads, axis=1)
        stackedHeads = np.transpose(stackedHeads, (0, 2, 1, 3))
        batchSize, sequenceLength, numberHeads, headDimension = stackedHeads.shape
        combinedAttention = stackedHeads.reshape(batchSize, sequenceLength, numberHeads * headDimension)
        return combinedAttention
        
    def outputProjectionLayer(self, combinedAttention):
        output = combinedAttention @ self.outputWeight + self.outputBias
        return output
    
    def forwardPropagation(self, inputEmbedding):
        Query, Key, Value = self.QKVLinearProjection(inputEmbedding)
        QHead, KHead, VHead = self.SplitIntoHeads(Query, Key, Value)
        combinedAttention = self.calculateAttention(QHead, KHead, VHead)
        output = self.outputProjectionLayer(combinedAttention)
        return output
    