from .layers import Layer

class Activationlayer(Layer):
	def __init__(self, input_shape, ouput_shape, activation, activation_prim):
		"""
		input_shape : đầu vào là 1 mảng
		output_shape : đầu ra là 1 mảng
		activation : hàm
		activation_prim : hàm
		"""
		self.input_shape = input_shape
		self.output_shape = ouput_shape
		self.activation = activation
		self.activation_prim = activation_prim # Đạo hàm

	def forward_propagation(self, inputt):
		self.input = inputt
		self.output = self.activation(inputt)
		return self.output

	def backward_propagation(self, ouput_error, learning_rate):
		return self.activation_prim(self.input) * ouput_error

