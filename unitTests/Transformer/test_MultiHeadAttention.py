from quacknet import MultiAttentionHeadLayer
import numpy as np

def createLayer():
    batchSize = 2
    sequenceLength = 4
    embedDimension = 8 
    numHeads = 2
    headDim = embedDimension // numHeads

    smallInput = np.array([
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [13.0, 14.0, 15.0, 16.0, 9.0, 10.0, 11.0, 12.0]
        ],
        [
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0]
        ]
    ])

    queryWeights = np.zeros((embedDimension, embedDimension))
    for i in range(numHeads):
        queryWeights[i*headDim:(i+1)*headDim, i*headDim:(i+1)*headDim] = np.eye(headDim)

    keyWeights = queryWeights.copy()
    valueWeights = queryWeights.copy()
    outputWeights = np.eye(embedDimension)
    outputBias = np.zeros((1, embedDimension))

    largeInput = np.ones((2, sequenceLength, embedDimension))

    layer = MultiAttentionHeadLayer(
        batchSize, sequenceLength, embedDimension, numHeads,
        queryWeights, keyWeights, valueWeights, outputWeights, outputBias
    )

    layer.QueryWeights = queryWeights
    layer.KeyWeights = keyWeights
    layer.ValueWeights = valueWeights
    layer.outputWeight = outputWeights
    layer.outputBias = outputBias

    return layer, smallInput, largeInput

def test_QKV():
    layer, _, input = createLayer()
    Q, K, V = layer.QKVLinearProjection(input)

    assert Q.shape == (2, 4, 8)
    assert K.shape == (2, 4, 8)
    assert V.shape == (2, 4, 8)

def test_SplitIntoHeads():
    layer, _, input = createLayer()
    Q, K, V = layer.QKVLinearProjection(input)
    qh, kh, vh = layer.SplitIntoHeads(Q, K, V)
    assert qh.shape == (2, 2, 4, 4)
    assert kh.shape == (2, 2, 4, 4)
    assert vh.shape == (2, 2, 4, 4)

def test_calculateAttention():
    layer, _, input = createLayer()
    Q, K, V = layer.QKVLinearProjection(input)
    qh, kh, vh = layer.SplitIntoHeads(Q, K, V)
    combinedAttention = layer.calculateAttention(qh, kh, vh)
    assert combinedAttention.shape == (2, 4, 8)

def test_outputProjectionLayer():
    layer, _, _ = createLayer()
    dummy_attention = np.ones((2, 4, 8))
    output = layer.outputProjectionLayer(dummy_attention)
    assert output.shape == (2, 4, 8)

def test_forwardPropagation():
    layer, _, input = createLayer()
    output = layer.forwardPropagation(input)
    assert output.shape == (2, 4, 8)

def test_QKV_numerical():
    layer, input, _ = createLayer()
    Q, K, V = layer.QKVLinearProjection(input)
    assert np.allclose(Q, input)
    assert np.allclose(K, input)
    assert np.allclose(V, input)

def test_attentionWeights():
    layer, input, _ = createLayer()
    Q, K, V = layer.QKVLinearProjection(input)
    Qh, Kh, Vh = layer.SplitIntoHeads(Q, K, V)
    _, weights = layer._calculateAttentionForOneHead(Qh[:, 0], Kh[:, 0], Vh[:, 0])
    summed = np.sum(weights, axis=-1)
    assert np.allclose(summed, np.ones_like(summed))

def test_forward():
    layer, input, _ = createLayer()
    output = layer.forwardPropagation(input)

    def softmax(x):
        x = np.array(x)
        ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return ex / np.sum(ex, axis=-1, keepdims=True)

    sqrt8 = np.sqrt(8)

    q0 = input[0, 0]
    q1 = input[0, 1] 

    q0_head0 = q0[0:4]
    q1_head0 = q1[0:4]
    k_head0 = input[0, :, 0:4]
    v_head0 = input[0, :, 0:4]

    q0_scores_h0 = (q0_head0 @ k_head0.T) / sqrt8
    weights_q0_h0 = softmax(q0_scores_h0)
    output_q0_h0 = weights_q0_h0 @ v_head0

    q1_scores_h0 = (q1_head0 @ k_head0.T) / sqrt8
    weights_q1_h0 = softmax(q1_scores_h0)
    output_q1_h0 = weights_q1_h0 @ v_head0

    q0_head1 = q0[4:8]
    q1_head1 = q1[4:8]
    k_head1 = input[0, :, 4:8]
    v_head1 = input[0, :, 4:8]

    q0_scores_h1 = (q0_head1 @ k_head1.T) / sqrt8
    weights_q0_h1 = softmax(q0_scores_h1)
    output_q0_h1 = weights_q0_h1 @ v_head1

    q1_scores_h1 = (q1_head1 @ k_head1.T) / sqrt8
    weights_q1_h1 = softmax(q1_scores_h1)
    output_q1_h1 = weights_q1_h1 @ v_head1

    combined_q0 = np.concatenate([output_q0_h0, output_q0_h1])
    combined_q1 = np.concatenate([output_q1_h0, output_q1_h1])

    expected_output = np.array([[combined_q0, combined_q1]])

    assert np.allclose(output[0, :2], expected_output[0])