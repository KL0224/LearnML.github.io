from .layers import Layer
import numpy as np

class FCLayer(Layer):
	def __init__(self, input_shape, output_shape):
		"""
		input_shape : (1, 3)
		output_shape : (1, 4)
		weight_shape : (3, 4)
		"""
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.weight = np.random.rand(input_shape[1], output_shape[1]) - 0.5
		self.bias = np.random.rand(1, output_shape[1]) - 0.5

	def forward_propagation(self, inputt):
		self.input = inputt
		self.output = np.dot(self.input, self.weight) + self.bias
		return self.output

	def backward_propagation(self, ouput_error, learning_rate):
		current_layer_error = np.dot(ouput_error, self.weight.T)
		dweight = np.dot(self.input.T, ouput_error) # (3,1)x(1,4)
		self.weight -= dweight * learning_rate
		self.bias -= learning_rate * ouput_error
		return current_layer_error




