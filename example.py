from classifier import classifier
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
input_data = preprocessing.scale(np.c_[mnist.data])
target_class = np.concatenate(np.c_[mnist.target],axis=0).astype(int)
training_values = np.random.randint(0,len(input_data)-1,5000)
validation_values = np.random.randint(0,len(input_data)-1,250)
training_inputs = input_data[training_values]
training_outputs = target_class[training_values]
validation_inputs = input_data[validation_values]
validation_outputs = target_class[validation_values]

input_dim = 28*28
output_dim = 10


# TODO develop unit testing and get comments on the current design
# to further develop the code

# Todo learn about different optimization approaches and the use of solvers like ADAMS
numberOfNeurons = [15,15,15]
# Todo Read on annotation in python 3.6
ann = classifier()
# Gradient descent parameters, play with these and see their effects
ann.configure_classifier(input_dim,output_dim,hidden_layers =numberOfNeurons,activation_function='relu',
                         batch_size=500)
ann.load_data(training_inputs,training_outputs)
model = ann.train_model(num_iterations=15000,epsilon = 1e-4,anneal="step", annealing_hyperparameters=[20,0.5])
predicted_outputs = ann.predict(validation_inputs)
error = sum((predicted_outputs-validation_outputs) != 0)
print(predicted_outputs-validation_outputs)
print(error)
