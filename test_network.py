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
net = network.Network([784, 33, 13])
#net.train(training_data, test_data)

net.load_parameters()

print(net.evaluate(test_data))

# # segment big image
# segments = get_segments('test_img.png')

# results = []
# for seg in segments:
#     results.append(np.argmax(net.feedforward(seg)))

# print results
