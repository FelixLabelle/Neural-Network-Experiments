import numpy as np
import random

class classifier:
    """This class implements a classifier with a variable number
     of hidden layers and neurons for each layer"""

    def relu(self, node_values):
        """Method implements RELU activation function"""
        return (node_values > 0) * node_values


    def relu_derivative(self, relu_vals):
        """Method implements RELU derivative"""
        return relu_vals > 0


    def tanh_derivative(self, tanh_vals):
        """Method implements tanh derivative"""
        return 1 - np.power(tanh_vals, 2)

    def sigmoid_derivative(self, sigmoid_vals):
        """Method implements sigmoid derivative"""
        return sigmoid_vals*(1-sigmoid_vals)

    def sigmoid(self, node_values):
        """Method implements sigmoid activation function"""
        # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        return 1/(1+np.exp(-node_values)) # Todo how to combat numerical issues?

    def __init__(self):
        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        self.X = np.array([])
        self.Y = np.array([])
        self.num_examples = 0


    def load_data(self, x, y):
        """Method loads training data"""
        self.X = x
        self.Y = y
        self.num_examples = len(self.X)  # training set size
        if self.batch_size == -1:
            self.batch_size = len(self.X)

    def configure_classifier(self, number_of_inputs, number_of_classes, hidden_layers = 5,
                               epsilon = 1e-5, reg_lambda = 1e-2, activation_function = "tanh",
                             batch_size = -1):
        """Sets training and neural network configurations"""
        # Gradient descent parameters, play with these and see their effects
        self.epsilon = epsilon
        self.reg_lambda = reg_lambda
        self.layers = [number_of_inputs] + hidden_layers + [number_of_classes] # rewrite this as a numpy array
        # find how to concatenate the middle one properly
        self.model = {}
        # creates a random matrix that is inputs by outputs
        self.weights = [np.random.randn(self.layers[i],
                                        self.layers[i+1]) / np.sqrt(self.layers[i]) for i in range(len(self.layers)-1)]
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)]
        self.a = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)]
        self.batch_size = batch_size

        # Consider passing function via arguments
        if activation_function == 'tanh':
            self.activation_function = np.tanh
            self.activation_derivative = self.tanh_derivative
        elif activation_function == 'relu':
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation_function == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        else:
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative



    def train_model(self, num_passes, print_loss=False):
        """This function calculates the cost function and backpropagates the error"""
        for i in range(0, num_passes):
            selection_array = random.sample(range(self.num_examples), self.batch_size)
            batch_input = self.X[selection_array] # make sure this gets the whole set
            batch_output = self.Y[selection_array]

            probs = self.__forward_prop__(batch_input)

            dW = [np.zeros(self.weights[i].shape) for i in range(len(self.layers)-1)]
            db = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]
            derivative = probs
            derivative[range(self.batch_size),batch_output] -= 1

            for i in range(len(self.layers) - 2,0,-1):
                dW[i] = (self.a[i-1].T).dot(derivative)
                dW[i] += self.reg_lambda *self.weights[i]
                db[i] = np.sum(derivative, axis=0, keepdims=True)
                derivative = derivative.dot(self.weights[i].T) * self.activation_derivative(self.a[i-1])

            dW[0] = np.dot(batch_input.T, derivative)
            dW[0] += self.reg_lambda * self.weights[0]
            db[0] = np.sum(derivative, axis=0)

            # Gradient descent parameter update
            # Consider implementing an annealing schedule here on labmda (the learning rate)
            for i in range(0, len(self.layers) - 1):
                self.weights[i] -= self.epsilon * dW[i]
                self.biases[i] -= self.epsilon * db[i]

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss()))

        #return [self.weights,self.biases]

    def predict(self, x):
        """ Predicts output values of a given dataset"""
        return np.argmax(self.__forward_prop__(x), axis=1)


    def calculate_loss(self):
        """ Evaluate error in the model """
        probs = self.__forward_prop__(self.X)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(self.num_examples), self.Y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        #data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.model['W1'])) + np.sum(np.square(self.model['W2'])) + np.sum(np.square(self.model['W3'])))
        # Add regularization this life back later
        return 1. / self.num_examples * data_loss

    def __forward_prop__(self, x):
        """ Evaluates forward propagation for a given set x"""
        z = x.dot(self.weights[0]) + self.biases[0]
        self.a[0] = self.activation_function(z)
        for i in range(1,len(self.layers) - 1):
            z = self.a[i-1].dot(self.weights[i]) + self.biases[i]
            self.a[i] = self.activation_function(z)
        # Todo Investigate on how to reduce risk of numerical errors in softmax
        exp_scores = np.exp(z)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
