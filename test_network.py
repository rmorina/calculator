"""
test_network.py
~~~~~~~~~~
"""

import neural_network as network
import data_loader as dl
import numpy as np
from segmentation import get_segments

# training
training_data, validation_data, test_data = dl.load_data()
net = network.Network([784, 30, 13])
net.stochastic_gradient_descent(training_data, 10, 10, 3.0, test_data=test_data)

# segment big image
segments = get_segments('test_img.png')

results = []
for seg in segments:
    results.append(np.argmax(net.feedforward(seg)))

print results
