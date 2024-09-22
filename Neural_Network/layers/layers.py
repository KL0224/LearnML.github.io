from abc import abstractmethod

class Layer:
	def __init__(self):
		self.input = None
		self.output = None
		self.input_shape = None # Lấy kích thước
		self.output_shape = None
		raise NotImplementedError

	@abstractmethod
	def input(self): # Hàm getter
		return self.input

	@abstractmethod
	def ouput(self):
		return self.output

	@abstractmethod
	def input_shape(self):
		return self.input_shape

	@abstractmethod
	def output_shape(self):
		return self.output_shape

	@abstractmethod
	def forward_propagation(self, inputt): # Lan truyền tuyến
		raise NotImplementedError

	@abstractmethod
	def backward_propagation(self, ouput_error, learning_rate):
		raise NotImplementedError # Hàm ảo