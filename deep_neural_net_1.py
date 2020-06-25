"""
Python script to construct a deep neural network. 

Takes the architecture -
1. Input size
2. Number of layers
3. Number of hidden units in each layer.
4. Number of output layers
5. Learning rate
6. Activation to use in each layer

Constructs a deep neural network, trains it on the input data set, and returns the parameters along with the error on the dataset.
"""

import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def help ():
    """
    Returns some helpful information on how to use
    """
    print('#'*5 + '-'*100 + '#'*5)
    print('\n\n')
    print('You will need to decide on the following hyperparameters to initialize your model - \n\n')
    print('The architecture of your model. \n\t Decide on the number of layers your network will have, and the number of units in each of these layers. \n')
    print('The activation functions to use in each layer. \n\t Most often you will want to use a ReLU activation for the hidden units and a sigmoid or softmax activation for the output.\n')
    print('The learning rate alpha of your model. \n\t Use a small number like 0.01, but not too small or gradient descent won\'t converge on the global minima. If you use too big a value, gradient descent can start to diverge.\n')
    print('The number of iterations of gradient descent you want to run. \n\t This model currently only supports gradient descent as an optimization algorithm, but I will be adding other optimizers like ADAM, RMS Prop, and Momentum soon.\n')
    print('\n')
    print('#'*5 + '-'*100 + '#'*5)
    print('\n\nIf you\'re working locally, make sure you have numpy and matplotlib installed.')
    print('\n')
    print('To build a network, simply call the model() function and pass it the parameters it needs. Have your input data and your output labels prepared and formatted how you want before you initialize.')
    print('Calling the model() function will begin training your given network on your given dataset for a default of 10,000 iterations\n')
    print('It will print the cost and the training accuracy of your model after it is done training.\n\n')
    print('#'*5 + '-'*100 + '#'*5)
    print('\n')
    

#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def sigmoid (Z):
    """
    Takes in a real number or matrix Z and applies the sigmoid function to it.
    """
    A = 1/(1+np.exp(-1*Z))
    return A

def relu (Z):
    """
    Takes in a real number or matrix Z and applies the (leaky) relu function to it.
    """
    A = max(0.01*Z, Z)
    return A

#NOTE TO SELF ---- THIS DEFINITION MIGHT CAUSE BROADCASTING PROBLEMS DURING RUNTIME
#NOTE 2.0 --- Probably fixed it.
def drelu (Z):
    """
    Takes in a real number Z and returns the derivative of the relu function at that value or matrix.
    Used for back propagation. Z will be the activation value of the last layer before the non-linearity is applied.
    """
    A = np.zeros(Z.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if Z[i,j] >= 0:
                A[i,j] = 1
            else:
                A[i,j] = 0.001
    return A

def softmax (Z):
    """
    Takes in a vector Z and returns its softmax activation.
    """
    temp = np.exp(Z)
    factor = np.sum(temp)
    return temp/factor

#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def initialize_parameters (architecture = []):
    """
    Initializes the parameters of the neural net.
    
    Takes in the architecture of the network as a list.
    Structure the list as the number of units in each layer, starting from the number of input features, to the number of output units.
    [10, 5, 5, 4, 3] means that there are 10 input features, 3 hidden layers with 5, 5, 4 hidden units respectively, and 3 output units (softmax regression)

    Returns a dictionary of keys W(i) for i from 1 to number of layers, and b(i) for the same i.
    """
    parameters = {}

    number_of_layers = len(architecture) - 1

    for i in range(1,number_of_layers+1):

        parameters['W' + str(i)] = np.random.randn(architecture[i], architecture[i-1])*0.01
        parameters['b' + str(i)] = np.zeros((architecture[i], 1))
    
    return parameters

#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def forward_propagation (X, parameters, num_layers, activation_func = []):
    """
    Performs forward propagation

    Takes in the activations of each layer and the parameters.
    activation_func is a list with as many elements (strings - either relu, softmax, or sigmoid) as number of layers (excluding the input layer)
    num_layers is the number of layers
    parameters is a dictionary containing all the W and b values for all the layers.
    X is the input

    Returns the final prediction y_hat and all the Z and A values as cache to use for backward propagation.
    """
    activations = {'A0': X}

    for i in range(len(1, num_layers+1)):
        W = parameters['W' + str(i)]
        b = parameters['b' + str(i)]

        activations['Z' + str(i)] = np.dot(W, activations['Z' + str(i-1)]) + b
        
        if activation_func[i-1].lower() == 'relu':
            activations['A' + str(i)] = relu(activations['Z' + str(i)])

        elif activation_func[i-1].lower() == 'softmax':
            activations['A' + str(i)] = softmax(activations['Z' + str(i)])

        elif activation_func[i-1].lower() == 'sigmoid':
            activations['A' + str(i)] = sigmoid(activations['Z' + str(i)])

    return activations['A' + str(num_layers)], activations

#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def calculate_cost (Y, prediction, activation_output = 'sigmoid'):
    """
    Calculates the cost.

    Takes in the output labels and the prediction from forward propagation, as well as the activation function of the output layer.
    Returns the cost.

    The inputs will be real numbers or row vectors of the same dimensions if the activation function of the last layer is sigmoid.
    The inputs will be row vectors or row matrices of the same dimensions if the activation function of the last layer is softmax.
    """
    m = Y.shape[1]

    if activation_output.lower() == 'sigmoid':
        cost = (1/m)*np.sum(Y*np.log(prediction) + (1-Y)*np.log(1-prediction))

    if activation_output.lower() == 'softmax':
        cost = (1/m)*np.sum(np.sum(Y*np.log(prediction), axis = 0, keepdims = True))

    return cost

#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def backward_propagation (X, Y, cache, number_of_layers):
    """
    Performs backward propagation.

    Takes in the inputs X, the corresponding labels Y, and the activation values stored as cache (returned by the second output for forward_propagation)
    Returns the gradients of all parameters in the grads dictionary.

    The cache is a dictionary containing the Z values and the A values for all layers.
    """
    m = Y.shape[1]
    last_layer = 'dZ' + str(number_of_layers)
    gradients = {last_layer: cache['A'+str(number_of_layers)]-Y}

    for i in reversed(range(1,number_of_layers)):
        gradients['dW' + str(i)] = (1/m)*np.dot(gradients['dZ'+str(i)], cache['A' + str(i-1)].T)
        gradients['db' + str(i)] = (1/m)*np.sum(gradients['dZ'+str(i)], axis = 1, keepdims = True)
        gradients['dZ' + str(i-1)] = np.dot(cache['W' + str(i)].T, gradients['dZ' + str(i)])*drelu(cache['Z' + str(i-1)])
    
    return gradients

#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def update_parameters (parameters, number_of_layers, gradients, alpha = 0.001):
    """
    Updates the parameters.

    Takes in the parameters themselves (as the parameters dictionary returned by the initialize_parameters function dictionary),
    the gradients (as the gradients dictionary returned by the backward_propagation function), and the learning rate alpha.
    Returns the updated parameters.
    """
    for i in range(1, number_of_layers+1):
        parameters['W' + str(i)] = parameters['W' + str(i)] - alpha*gradients['dW' + str(i)]
        parameters['b' + str(i)] = parameters['b' + str(i)] - alpha*gradients['db' + str(i)]
    
    return parameters

#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def calculate_accuracy (X, Y, parameters, number_of_layers, activation_functions):
    """
    Runs forward propagation on all examples and returns the accuracy of the model

    Takes in X and Y, the parameters, number of layers, and the activation functions list. Meant to be run inside the model() function definition.
    """
    m = Y.shape[1]
    correct_count = 0

    for i in range(m):
        X_in = X[:, i]
        Y_out = Y[:, i]

        pred = forward_propagation(X_in, parameters, number_of_layers, activation_functions)

        if pred == Y_out:
            correct_count += 1
    
    accuracy = correct_count*100/m
    return accuracy

#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def model (X, Y, architecture, activation_functions, learning_rate = 0.001, print_cost = True, number_of_iterations = 10000):
    """
    Takes in the training set, the labels, and all the required parameters and trains the defined model for the given number of iterations.
    Prints the cost if print_cost is true.
    """
    costs = []
    number_of_layers = len(architecture) - 1

    parameters = initialize_parameters(architecture)

    for i in range(number_of_iterations):

        prediction, cache = forward_propagation(X, parameters, number_of_layers, activation_functions)
        cost = calculate_cost(Y, prediction, activation_functions[-1])

        costs.append(cost)
        gradients = backward_propagation(X, Y, cache, number_of_layers)

        parameters = update_parameters(parameters, number_of_layers, gradients, alpha=0.01)

        if print_cost and i%100 == 0:
            print(f'Cost after iteration {i} = {cost}')

    plt.plot(costs)
    plt.xlabel('Iterations (in hundereds')
    plt.ylabel('Cost')
    plt.title(f'Learning rate = {learning_rate}')
    plt.show()

    print(f'Ran {number_of_iterations} iterations. Returning parameters now.')
    print(f'The final cost of the model was: {costs[-1]}')
    print(f'Training accuracy was: {calculate_accuracy(X, Y, parameters, number_of_layers, activation_functions)}')

    return parameters
