"""All the optimization methods go here.

"""

from __future__ import division, print_function, absolute_import
import random
import numpy as np
import copy


class SGD(object):
    """Mini-batch stochastic gradient descent.

    Attributes:
        learning_rate(float): the learning rate to use.
        batch_size(int): the number of samples in a mini-batch.

    """

    def __init__(self, learning_rate, batch_size):
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size

    def __has_parameters(self, layer):
        return hasattr(layer, "W")

    def compute_gradient(self, x, y, graph, loss):
        """ Compute the gradients of network parameters (weights and biases)
        using backpropagation.

        Args:
            x(np.array): the input to the network.
            y(np.array): the ground truth of the input.
            graph(obj): the network structure.
            loss(obj): the loss function for the network.

        Returns:
            dv_Ws(list): a list of gradients of the weights.
            dv_bs(list): a list of gradients of the biases.

        """

        # TODO: Backpropagation code
        leng=len(graph.config)
        layer_all=[layer for layer in graph]
        
        out=[x]
        y_copy=copy.copy(x)
        #forward
        for layer in graph[:leng]:
            y_copy=layer.forward(y_copy)
            out.append(y_copy)
        
        dv_y=loss.backward(y_copy,y)
        dv_Ws=[]
        dv_bs=[]
        #sigmoid odd index
        dy= layer_all[3].backward(out[3], dv_y)
        #fully connected even index
        dv_x, dv_W, dv_b= layer_all[2].backward(out[2],dy)
        dv_Ws.append(dv_W)
        dv_bs.append(dv_b)       
        #the rest
        dy=layer_all[1].backward(out[1], dv_x)
        dv_x, dv_W, dv_b= layer_all[0].backward(out[0],dy)
        dv_Ws.append(dv_W)
        dv_bs.append(dv_b)        
        
        #reverse            
        return dv_Ws[::-1],dv_bs[::-1]

        pass

    def optimize(self, graph, loss, training_data):
        """ Perform SGD on the network defined by 'graph' using
        'training_data'.

        Args:
            graph(obj): a 'Graph' object that defines the structure of a
                neural network.
            loss(obj): the loss function for the network.
            training_data(list): a list of tuples ``(x, y)`` representing the
                training inputs and the desired outputs.

        """

        # Network parameters
        Ws = [layer.W for layer in graph if self.__has_parameters(layer)]
        bs = [layer.b for layer in graph if self.__has_parameters(layer)]

        # Shuffle the data to make sure samples in each batch are not
        # correlated
        random.shuffle(training_data)
        n = len(training_data)

        batches = [
            training_data[k:k + self.batch_size]
            for k in xrange(0, n, self.batch_size)
        ]
        
        bat=0
        # TODO: SGD code
        for data in batches:
            bat+=1
            print(bat)
            for data2 in data:
                dv_Ws,dv_bs=self.compute_gradient(data2[0],data2[1],graph,loss)
                layers=[layer for layer in graph]
                for i in range(len(layers)):
                    if i%2==0:
                        layers[i].W=layers[i].W-np.multiply(self.learning_rate,dv_Ws[int(i/2)])
                        layers[i].b=layers[i].b-np.multiply(self.learning_rate,dv_bs[int(i/2)])
        pass
