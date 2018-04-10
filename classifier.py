import numpy as np
import random
from sklearn import preprocessing

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

    # This method introduces an error, the intial learning rate will be too small by a factor of step_factor
    def step_annealing(self,epsilon,current_iteration,hyperparameters):
        """Reduces the learning rate by step_factor for in a step like fashion"""
        [reduction_interval, step_factor] = hyperparameters
        num_epochs = np.floor((self.batch_size*current_iteration)/self.num_examples)
        return epsilon * step_factor**np.ceil(reduction_interval/num_epochs)

    def fixed_rate_annealing(self,epsilon,current_iteration,hyperparameters):
        """Reduces the learning rate by step_factor using an exponential function"""
        k = hyperparameters[0]
        return epsilon * np.exp(-k*current_iteration)

    def exponential_annealing(self,epsilon,current_iteration,hyperparameters):
        """Reduces the learning rate by step_factor using a fixed rate"""
        k = hyperparameters[0]
        return epsilon / (1+k*current_iteration)

    def fixed_learning_rate(self,epsilon,current_iteration,hyperparameters):
        """Method allows for generalization by implementing annealing interface"""
        return epsilon

    def __init__(self):
        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        self.X = np.array([])
        self.Y = np.array([])
        self.num_examples = 0

    # TODO implement normalization of your data
    def load_data(self, x, y, normalize_data = False):
        """Method loads training data"""
        self.X = x
        self.Y = y
        self.num_examples = len(self.X)  # training set size
        if self.batch_size == -1:
            self.batch_size = len(self.X)

        if normalize_data:
            self.X = preprocessing.scale(self.X)

    # Todo add choice of classifier, as an external class
    def configure_classifier(self, number_of_inputs, number_of_classes, hidden_layers = 5,
                               activation_function = "tanh",batch_size = -1):
        """Sets training and neural network configurations"""
        self.layers = [number_of_inputs] + hidden_layers + [number_of_classes] # rewrite this as a numpy array
        self.model = {}
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

    def train_model(self, num_iterations, epsilon = 1e-5, reg_lambda = 1e-2,
                    print_loss=False, anneal = "default",annealing_hyperparameters = [1,1]):
        """This function calculates the cost function and backpropagates the error"""
        if anneal == "step":
            learning_function = self.step_annealing
        elif anneal == "exponential":
            learning_function = self.exponential_annealing
        elif anneal == "fixed":
            learning_function = self.fixed_rate_annealing
        else:
            learning_function = self.fixed_learning_rate

        for i in range(0, num_iterations):
            #random_indices = np.arange(self.num_examples)
            #np.random.shuffle(random_indices)
            #selection_array = random_indices[0:self.batch_size]
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
                dW[i] += reg_lambda *self.weights[i]
                db[i] = np.sum(derivative, axis=0, keepdims=True)
                derivative = derivative.dot(self.weights[i].T) * self.activation_derivative(self.a[i-1])

            dW[0] = np.dot(batch_input.T, derivative)
            dW[0] += reg_lambda * self.weights[0]
            db[0] = np.sum(derivative, axis=0)

            # Gradient descent parameter update
            learning_rate = learning_function(epsilon, i, annealing_hyperparameters)
            for i in range(0, len(self.layers) - 1):
                self.weights[i] -= learning_rate * dW[i]
                self.biases[i] -= learning_rate * db[i]

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss()))

    def predict(self, x):
        """ Predicts output values of a given dataset"""
        return np.argmax(self.__forward_prop__(x), axis=1)


    def calculate_loss(self):
        """ Evaluate error in the model """
        probs = self.__forward_prop__(self.X)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(self.num_examples), self.Y])
        data_loss = np.sum(corect_logprobs)
        # Todo: add regulatization term to loss
        return 1. / self.num_examples * data_loss

    def __forward_prop__(self, x):
        """ Evaluates forward propagation for a given set x"""
        z = x.dot(self.weights[0]) + self.biases[0]
        self.a[0] = self.activation_function(z)
        for i in range(1,len(self.layers) - 1):
            z = self.a[i-1].dot(self.weights[i]) + self.biases[i]
            self.a[i] = self.activation_function(z)
        # Todo: Investigate on how to reduce risk of numerical errors in softmax
        exp_scores = np.exp(z)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
