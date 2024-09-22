from network.network import Network
from layers.FCLayer import FCLayer
from layers.activation_layer import Activationlayer
import numpy as np

def relu(z):
	"""
	z : numpy array
	return : 0 if z <= 0 hoặc z nếu z > 0
	"""
	return np.maximum(0, z)

def relu_prime(z):
	"""
	z : numpy array
	return : 0 if z < 0 hoặc 1 nếu z > 0
	"""
	z[z < 0] = 0
	z[z > 0] = 1
	return z

# Hàm cost
def loss(y_true, y_predict):
	return 0.5 * (y_predict - y_true)**2

def loss_prime(y_true, y_predict):
	return y_predict - y_true

x_train = np.array([[[0, 0]], [[1, 0]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]]) 

net = Network()
net.add(FCLayer((1, 2), (1, 3)))
net.add(Activationlayer((1, 3), (1, 3), relu, relu_prime))
net.add(FCLayer((1, 3), (1, 1)))
net.add(Activationlayer((1, 1), (1, 1), relu, relu_prime))
net.setup_loss(loss, loss_prime)
net.fit(x_train, y_train, learning_rate = 0.01, epochs = 1000)
out = net.predict([[0, 1]])
print(out)