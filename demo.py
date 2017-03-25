"""This is a simple example for training a neural network. This example
demonstrates some usage of the modules. You may use this code to test your
implementation.

"""

import mnist_loader
import numpy as np
from graph import Graph
from loss import Euclidean
from network import Network
from optimization import SGD
import time

# Load the MNIST dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# The network definition of a neural network
graph_config = [
    ("FullyConnected", {"shape": (30, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
]

graph = Graph(graph_config)
loss = Euclidean()
optimizer = SGD(1.2, 10)  # learning rate 1.2, batch size 10
network = Network(graph)

# Train the network for 1 epoch
start_time = time.time()
network.train(training_data, 1, loss, optimizer, test_data)

# Test a handwritten digit image
x, y = test_data[800]
y_pred = np.argmax(network.inference(x))
print("The image is {}. Your prediction is {}.".format(y, y_pred))
print("--- %s seconds ---" % (time.time() - start_time))
