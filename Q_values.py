import numpy as np


def Q_values(x, W1, W2, bias_W1, bias_W2):

    """
    FILL THE CODE
    Compute the Q values as ouput of the neural network.
    W1 and bias_W1 refer to the first layer
    W2 and bias_W2 refer to the second layer
    Use rectified linear units
    The output vectors of this function are Q and out1
    Q is the ouptut of the neural network: the Q values
    out1 contains the activation of the nodes of the first layer
    there are othere possibilities, these are our suggestions
    YOUR CODE STARTS HERE
    """

    def ReLU(x): # activation function
        return x * (x > 0)

    hiddenLayerInput1=np.dot(x,W1)
    hiddenLayerInput=hiddenLayerInput1 + bias_W1
    hiddenLayerActivations = ReLU(hiddenLayerInput)

    outputLayerInput1=np.dot(hiddenLayerActivations,W2)
    outputLayerInput= outputLayerInput1+ bias_W2
    Q = ReLU(outputLayerInput)

    # YOUR CODE ENDS HERE

    return Q, outputLayerInput, hiddenLayerActivations, hiddenLayerInput
