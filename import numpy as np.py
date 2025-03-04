import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


inputs = np.array([[0.05, 0.10]])  
expected_output = np.array([[0.01, 0.99]])


np.random.seed(1)
weights_input_hidden = np.array([[0.15, 0.20], [0.25, 0.30]])
weights_hidden_output = np.array([[0.40, 0.45], [0.50, 0.55]])
bias_hidden = np.array([[0.35, 0.35]])
bias_output = np.array([[0.60, 0.60]])


alpha = 0.5

hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
output_layer_output = sigmoid(output_layer_input)

delta_output = (expected_output - output_layer_output) * sigmoid_derivative(output_layer_output)

delta_hidden = np.dot(delta_output, weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

weights_hidden_output += alpha * np.dot(hidden_layer_output.T, delta_output)
weights_input_hidden += alpha * np.dot(inputs.T, delta_hidden)
bias_output += alpha * delta_output
bias_hidden += alpha * delta_hidden

print("Updated Weights (Input to Hidden):")
print(weights_input_hidden)
print("Updated Weights (Hidden to Output):")
print(weights_hidden_output)
print("Updated Biases (Hidden Layer):")
print(bias_hidden)
print("Updated Biases (Output Layer):")
print(bias_output)