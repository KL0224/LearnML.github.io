from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split # Chia dataset 75% để training và 25% còn lại để kiểm tra
from sklearn.tree import DecisionTreeClassifier
import numpy as np
iris_dataset = load_iris()
#print(iris_dataset.data)
#print(iris_dataset.target)
#print(len(iris_dataset.target))
# x thuộc data, y thuộc target
x_train, x_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, random_state = 0)
model = DecisionTreeClassifier()
my_model = model.fit(x_train, y_train)
x_new = np.array([[1.0, 3.0, 7.9, 6.2]])
#print(my_model.predict(x_new))
print(my_model.score(x_test, y_test))