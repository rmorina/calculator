"""
test_network.py
~~~~~~~~~~
"""

import neural_network as network
import data_loader as dl

training_data, validation_data, test_data = dl.load_data()
net = network.Network([784, 30, 13])
net.stochastic_gradient_descent(training_data, 10, 10, 3.0, test_data=test_data)