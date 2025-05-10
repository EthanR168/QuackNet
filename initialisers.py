import math, random
import numpy as np
from activationFunctions import relu, sigmoid

class Initialisers:
    def createWeightsAndBiases(self):
        #weights are in [number of layers][size of current layer][size of next layer]
        for i in range(1, len(self.layers)):
            currSize = self.layers[i][0]
            lastSize = self.layers[i - 1][0]
            actFunc = self.layers[i][1]

            if(actFunc == relu):
                bounds =  math.sqrt(2 / lastSize) # He initialisation
            elif(actFunc == sigmoid):
                bounds = math.sqrt(6/ (lastSize + currSize)) # Xavier initialisation
            else:
                bounds = 1
                
            currW = []
            for _ in range(currSize):
                w = []
                for _ in range(lastSize):
                    w.append(random.gauss(0, bounds)) 
                currW.append(w)
            self.weights.append(currW)

            b = []
            for _ in range(currSize):
                b.append(0)
            self.biases.append(b)
        