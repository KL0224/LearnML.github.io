import pandas as pd
import numpy as np
import matplotlib.pyplot as mat

data = pd.read_csv("data_classification.csv", header = None)
#print(data)
#print(data.values)

true_x = []
true_y = []
false_x = []
false_y = []
for item in data.values:
	if item[2] == 1:
		true_x.append(item[0])
		true_y.append(item[1])
	else:
		false_x.append(item[0])
		false_y.append(item[1])

mat.scatter(true_x, true_y, marker = 'o', c = 'b')
mat.scatter(false_x, false_y, marker = 'o', c = 'r')
#mat.show()

# Hàm sigmoid
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# Tìm ranh giới
def phanchia(p):
	if p < 0.5: 
		return 0
	else: 
		return 1

# Dự đoán
def predict(features, weights):
	z = np.dot(features, weights)
	return sigmoid(z)

# Hàm chi phí để tối ưu chi phí
def cost_function(features, labels, weights):
	"""
	param features : [100*3] # dữ liệu
	param lable : [100*1] => 1 hoặc 0
	param weights : [3 * 1]
	retrun chi phi
	y^T => Ma trận chuyển vị chuyển ma trận cột sang hàng
	"""

	n = len(labels)
	predictions = predict(features, weights) # chứa [0.6, 0.4, ...]
	cost_class1 = -lables * np.log(predictions)
	cost_class2 = -(1 - lables) * np.log(1 - predictions) 
	cost = cost_class1 + cost_class2
	return cost.sum() / n

def update_weight(features, labels, weights, learning_rate):
	"""
	param features : [100*3] # dữ liệu
	param lable : [100*1] => 1 hoặc 0
	param weights : [3 * 1]
	param learning_rate : float
	retrun new_weight : float
	"""

	n = len(labels)
	predictions = predict(features, weights)
	gd = np.dot(features.T, (predictions - lables))
	gd /= np
	gd *= learning_rate
	weights = weights - gd
	return weights

def train(features, labels, weights, learning_rate, iter):
	cost_his = []
	for i in range(iter):
		weights = update_weight(features, labels, weights, learning_rate)
		cost = cost_function(features, labels, weights)
		cost_his.append(cost)
	return weights, cost_his

weights, cost_his = train()

# Bài tập viết dụ đoán.

