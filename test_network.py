"""
test_network.py
~~~~~~~~~~
"""
import neural_network as network
import data_loader as dl
import numpy as np
from segmentation import get_segments

net = network.Network([784, 30, 13])
net.load_parameters()


# # segment big image
# segments = get_segments('test_img.png')

# results = []
# for seg in segments:
#     results.append(np.argmax(net.feedforward(seg)))

# print results
