# Disclaimer: This code was primarily developed based on the tutorial at 
# neuralnetworksanddeeplearning.com with modification suited for our project.
"""
neural_network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation. 
"""

import numpy as np
import random

class Network(object):

    def __init__(self, sizes):
        """sizes is a list whose length indicates the number of layers the
        network will use and the individual elements indicate the number of 
        elements in each layer of the network"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        # each layer has its bias. Notice layer one is the input layer and
        # hence it inherently does not have a bias
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = []
        for i in xrange(len(sizes) - 1):
            cur_layer_size = sizes[i]
            next_layer_size = sizes[i + 1]
            self.weights += [np.random.randn(next_layer_size, cur_layer_size)]

    def feedforward(self, a):
        """Returns the output of the network by propagating forward the output 
        of each individual layer. a is the vector input to the network"""
        for b,w in zip(self.biases, self.weights):
            # a remains a vector and indicates the input to each layer in the
            # network
            a = sigmoid(np.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(self, training_data, epochs, 
                                    mini_batch_size, eta, test_data=None):
        """training_data - a list of tuples (x, y) where x is the input and y 
        is the desired output. In our case x for example is a 784 dimensional
        vector (since our training images are 28x28) and y is a 13 dimensional 
        vector (since we have 10 digits and 3 arithmetic operations: +, -, *
            epochs - number of epochs to train for
            mini_batch_size - size of the mini-batches to use when sampling
            eta - learning rate"""
        if test_data: test_len = len(test_data)
        training_len = len(training_data)
        for i in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = []
            for k in xrange(0, training_len, mini_batch_size):
                mini_batches += [training_data[k : k + mini_batch_size]]
            for mini_batch in mini_batches:
                # apply a single step of the gradient descent using just the
                # taining data in the mini batch
                self.update_parameters(mini_batch, eta)
            if test_data:
                print ("Epoch %d: %d / %d" % 
                    (i, self.evaluate(test_data), test_len))
            else:
                print("Epoch %d complete" % i)

    def update_parameters(self, mini_batch, eta):
        """update the network's weight and biases by applying gradient descent
        using backpropagation to a signle mini batch.
           mini_batch - list of tuples (x, y), where x is the input and y the 
        desired output of the network.
           eta - learning rate"""
        batch_size = len(mini_batch)
        # C as in standard notation stands for gradient
        C_b = [np.zeros(b.shape) for b in self.biases]
        C_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # C_x_b gradient with respect to the biases in each level for the
            # given image
            C_x_b, C_x_w = self.backprop(x, y)
            C_b = [cb + cxb for cb, cxb in zip(C_b, C_x_b)]
            C_w = [cw + cxw for cw, cxw in zip(C_w, C_x_w)]
        self.weights = [w - (eta/batch_size)*cw for w, cw in zip(self.weights, C_w)]
        self.biases = [b - (eta/batch_size)*cb for b, cb in zip(self.biases, C_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Helper functions
def sigmoid(z):
    """z is a vector that was computed w*a + b where w is a matrix that
    describes the relationship between the layer n and n + 1, a is the
    input from layer n and b is the bias. This function applys the sigmoid
    function elementwise."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))




