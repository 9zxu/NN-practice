# https://medium.com/@hadican/how-to-build-a-simple-artificial-neural-network-ann-a064939f940b

import numpy as np
from decimal import Decimal, ROUND_HALF_UP

class ANN:
    def __init__(self):
        self.weights = 2*np.random.random((4,1))-1
    # 

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # forward pass
    def forward(self, inputs):
        # e.g. [1,0,0,1]
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output
    
    def train(self, training_inputs, training_outputs, number_of_iterations):
        for iteration in range(number_of_iterations):
            output = self.forward(training_inputs)
            # calculate loss
            error = training_outputs - output 
             # optimize
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.weights += adjustments



if __name__ == "__main__":
    model = ANN()
    print("Randomly Generated Weights:", model.weights)

    training_inputs = np.array([[1, 0, 0, 1],
                                [0, 1, 1, 0],
                                [1, 0, 1, 0],
                                [0, 0, 0, 1]])

    training_outputs = np.array([[1],
                                 [0],
                                 [1],
                                 [0]])
    
    model.train(training_inputs, training_outputs, 10000)

    print("Weights After Training:")
    print(model.weights)

    input = np.array([1, 0, 1, 0])
    print("testing:", input)

    output = model.forward(input)  # e.g. 0.9989834
    result = Decimal(output[0]).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    # quantize 方法可以將一個浮點數，依指示的有效位數(number of significant digits) 進行約整(rounding，或稱捨入)
    print("The answer for {} is: {}".format(input, result))