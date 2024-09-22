import pandas as pd
import matplotlib.pyplot as mat

dataframe = pd.read_csv("Advertising.csv")
X = dataframe.values[: , 2] # Lấy cột Radio
Y = dataframe.values[: , 4] # Lấy cột Sales
# mat.scatter(X, Y, marker = 'o')
# mat.show()

def predict(new_radio, weight, bias):
	return weight * new_radio + bias

def cost_function(X, Y, weight, bias):
	n = len(X)
	sum_error = 0
	for i in range(n):
		sum_error += (Y[i] - (weight * X[i] + bias))**2
	return sum_error / n

def update_weight(X, Y, weight, bias, learning_rate):
	n = len(X)
	weight_temp = 0.0
	bias_temp = 0.0
	for i in range(n):
		weight_temp += -2 * X[i] * (Y[i] - (weight * X[i] + bias))
		bias_temp += -2 * (Y[i] - (weight * X[i] + bias))
	weight -= (weight_temp / n) * learning_rate
	bias -= (bias_temp / n) * learning_rate
	return weight, bias

def train(X, Y, weight, bias, learning_rate, iter):
	cost_his = []
	for i in range(iter):
		weight, bias = update_weight(X, Y, weight, bias, learning_rate)
		cost = cost_function(X, Y, weight, bias)
		cost_his.append(cost)
	return weight, bias, cost_his

weight, bias, cost = train(X, Y, 0.03, 0.0014, 0.001, 60)
print(weight, bias, cost, sep = '\n')
print("Gia tri du doan:", predict(19, weight, bias))

solanlap = [i for i in range(60)]
mat.plot(solanlap, cost)
mat.show()