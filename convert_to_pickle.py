"""
A library to serialize the arithmetic symbols images
"""

import cPickle
import gzip

# Third-party libraries
import numpy as np
import os
from PIL import Image

def readFile(path):
    with open(path, "rt") as f:
        return f.read()

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

def load_data(path):
    symbols_map = {'+':10, '-':11, 'x':12}
    images = None
    labels = np.array([])
    for symbol in symbols_map:
        symbol_path = path + symbol
        symbol_digit = symbols_map[symbol]
        print(symbol_path)
        for filename in os.listdir(symbol_path):
            full_path = symbol_path + os.sep + filename
            im = np.asarray(Image.open(full_path))
            length, width = im.shape
            im = np.reshape(im, (1, width*length))
            im = (255 - im)/255.0
            if images is None:
                images = im
            else:
                images = np.append(images, im, axis=0)
            labels = np.append(labels, symbol_digit)
    return (images, labels)


def convert_to_pickle():
    base_training_path = 'data/symbols/symbols_training/'
    base_validation_path = 'data/symbols/symbols_validation/'
    training_data = load_data(base_training_path)
    validation_data = load_data(base_validation_path)
    # storing the validation data twice as the expected format for the pickle
    # files utilized in the neural network is training_data, validation_data,
    # test_data. Since the overall size of the data for arithmetic symbols is
    # much smaller than that of the digits, we only partition this data set into
    # two subsets so as to have a large enough number of training samples. 
    all_data = (training_data, validation_data, validation_data)
    cPickle.dump(all_data, open( "data/symbols.pkl", "wb" ))

