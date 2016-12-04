"""
A library to load the MNIST image data and the arithmetic operations.
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

from PIL import Image

def load_raw_data(path):
    training_data, validation_data, test_data = cPickle.load(open(path, "rb" ))
    return (training_data, validation_data, test_data)

def reformat_data(tr_digits, tr_symbols, va_digits, va_symbols, te_digits,
        te_symbols):
    """
        Combines the data set for digits and arithmetic symbols into one data
        sets and formats it so that it is convinient for the neural netowk to
        train and test. In particular the final ``training_data`` returned
        is a list containing``(x, y)``. where ``x`` is a 784-dimensional
        numpy.ndarray containing the input image and ``y`` is a 13-dimensional
        numpy.ndarray representing the unit vector corresponding to the
        correct digit for ``x`` or one of the '+', '-', 'x' which are
        represented as 10, 11, or 12, respectively.

        ``validation_data`` and ``test_data`` are lists also containing
        tuples ``(x, y)``, where x is again a 784-dimensional
        numpy.ndarry containing the input image, but ``y`` is the
        corresponding classification, i.e., simply the digit values (0,...,12)
        corresponding to ``x``.

        The difference in the format of training_data and validation_data allow
        turns to simply be a convinience when using the neural network.
    """
    training_digit_images = [np.reshape(x, (784, 1)) for x in tr_digits[0]]
    training_symbol_images = [np.reshape(x, (784, 1)) for x in tr_symbols[0]]
    training_inputs = training_digit_images + training_symbol_images

    training_digit_labels = [vectorized_result(y) for y in tr_digits[1]]
    training_symbol_labels = [vectorized_result(y) for y in tr_symbols[1]]
    training_results = training_digit_labels + training_symbol_labels

    training_data = zip(training_inputs, training_results)

    validation_digit_images = [np.reshape(x, (784, 1)) for x in va_digits[0]]
    validation_symbol_images = [np.reshape(x, (784, 1)) for x in va_symbols[0]]
    validation_inputs = validation_digit_images + validation_symbol_images

    validation_results = np.append(va_digits[1], va_symbols[1])

    validation_data = zip(validation_inputs, validation_results)

    test_digit_images = [np.reshape(x, (784, 1)) for x in te_digits[0]]
    test_symbol_images = [np.reshape(x, (784, 1)) for x in te_symbols[0]]
    test_inputs = test_digit_images + test_symbol_images

    test_results = np.append(te_digits[1], te_symbols[1])

    test_data = zip(test_inputs, test_results)

    return (training_data, validation_data, test_data)

# only load the training digits. Temporary until we have more training images
# for arithmetic symbols
def reformat_data_2(tr_digits, va_digits, te_digits):
    training_digit_images = [np.reshape(x, (784, 1)) for x in tr_digits[0]]
    training_inputs = training_digit_images

    training_digit_labels = [vectorized_result(y) for y in tr_digits[1]]
    training_results = training_digit_labels

    training_data = zip(training_inputs, training_results)

    validation_digit_images = [np.reshape(x, (784, 1)) for x in va_digits[0]]
    validation_inputs = validation_digit_images
    validation_results = va_digits[1]

    validation_data = zip(validation_inputs, validation_results)

    test_digit_images = [np.reshape(x, (784, 1)) for x in te_digits[0]]
    test_inputs = test_digit_images

    test_results = te_digits[1]

    test_data = zip(test_inputs, test_results)

    return (training_data, validation_data, test_data)

def load_data():
    digit_data_path = 'data/mnist.pkl'
    #symbol_data_path = 'data/symbols.pkl'
    tr_digits, va_digits, te_digits = load_raw_data(digit_data_path)

    #tr_symbols, va_symbols, te_symbols = load_raw_data(symbol_data_path)
    training_data, validation_data, test_data = reformat_data_2(tr_digits,
        va_digits, te_digits)

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 13-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) or arithmetic symbol (+ - x) into a corresponding desired output
    from the neural network."""
    e = np.zeros((13, 1))
    if j in range(10):
        e[j] = 1.0
    elif j == '+':
        e[10] = 1.0
    elif j == '-':
        e[11] = 1.0
    elif j == 'x':
        e[12] = 1.0
    return e
