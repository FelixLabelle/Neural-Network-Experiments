from classifier import neural_network
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata

# Retrieve MNIST data and prep valid/test set
size_training_data = 5500
size_validation_data = 500
mnist = fetch_mldata('MNIST original')
input_data = preprocessing.scale(np.c_[mnist.data])
target_class = np.concatenate(np.c_[mnist.target],axis=0).astype(int)
random_indices = np.arange(len(input_data))
np.random.shuffle(random_indices)
training_values = random_indices[0:size_training_data-1]
validation_values = random_indices[size_training_data:size_validation_data+size_training_data-1]
training_inputs = input_data[training_values]
training_outputs = target_class[training_values]
validation_inputs = input_data[validation_values]
validation_outputs = target_class[validation_values]

# Define input and output dimensions
input_dim = training_inputs.shape[1]
output_dim = 10


# TODO develop unit testing and get comments on the current design
# to further develop the code

# Todo learn about different optimization approaches and the use of solvers like ADAMS
# Todo Implement hyperparameter optimization
numberOfNeurons = [output_dim]
# Todo Read on annotation in python 3.6
ann = neural_network()
# Gradient descent parameters, play with these and see their effects
ann.configure_classifier(input_dim,output_dim,hidden_layers =numberOfNeurons,activation_function='relu',
                         batch_size=500,epsilon = 1e-4)
ann.load_data(training_inputs,training_outputs)
model = ann.train_model(num_iterations=15000)
predicted_outputs = ann.predict(validation_inputs)
error = sum((predicted_outputs-validation_outputs) != 0)
print(predicted_outputs-validation_outputs)
print("The error rate is " + str(error*100/size_validation_data) +"%")
