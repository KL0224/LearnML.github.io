

class Network:
	def __init__(self):
		self.layers = []
		self.loss = None # hàm
		self.loss_prime = None # hàm

	def add(self, layer):
		self.layers.append(layer)

	def setup_loss(self, loss, loss_prime):
		self.loss = loss
		self.loss_prime = loss_prime

	def predict(self, inputt):
		"""
		inputt : [[1, 3]] => 1
		return : kết quả dữ đoán
		"""
		result = []
		n = len(inputt)
		for i in range(n):
			output = inputt[i]
			for layer in self.layers:
				output = layer.forward_propagation(output)
			result.append(output)
		return result

	def fit(self, x_train, y_train, learning_rate, epochs):
		n = len(x_train)
		for i in range(epochs):
			error = 0
			for j in range(n):
				# Lan truyền tuyến
				output = x_train[j]
				for layer in self.layers:
					output = layer.forward_propagation(output)
				# Tính lỗi của từng sample
				error += self.loss(y_train[j], output)

				# Lan truyền ngược
				errorb = self.loss_prime(y_train[j], output)
				for layer in reversed(self.layers):
					errorb = layer.backward_propagation(errorb, learning_rate)

				error /= n
				print("epoch : %d/%d error = %f"%(i, epochs, error))



