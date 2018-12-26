import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

'''
### draw raw data
print(df.label.value_counts())
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
'''

data = np. array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])


"""
### perceptron model
class Model:
	def __init__(self):
		self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
		self.b = 0
		self.l_rate = 0.1
	def sign(self, x, w, b):
		y = np.dot(w, x) + b
		return y
	# stochastic gradient descent
	def fit(self, X_train, y_train):
		is_wrong = False
		iter_n = 0
		while not is_wrong:
			wrong_count = 0
			iter_n += 1
			for d in range(len(X_train)):
				X = X_train[d]
				y = y_train[d]
				if y * self.sign(X, self.w, self.b) <= 0:
					self.w = self.w + self.l_rate * np.dot(X, y)
					self.b = self.b + self.l_rate * y
					wrong_count += 1					
					# show the change of iter
					#plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
					#plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
					#plt.xlabel('sepal length')
					#plt.ylabel('sepal width')
					#plt.legend()
					x_points = np.linspace(4, 7, 10)
					y_ = -(self.w[0] * x_points + self.b) / self.w[1]
					plt.plot(x_points, y_)
					plt.show()
			if wrong_count == 0:
				is_wrong = True
		return 'Perceptron Model!', iter_n
	def score(self):
		pass
"""


### perceptron dual form
class dualModel:
	def __init__(self):
		self.w = np.zeros(len(data[0]) - 1, dtype=np.float32)
		self.alpha = np.ones(len(data), dtype=np.float32)
		self.b = 0
		self.ita = 1
		self.gram = []

	def sign(self, X, y, alpha, n):
		w = 0
		#gram = self.gram_mat(X)
		for d in range(len(data)):
			w = w + alpha[d] * y[d] * self.gram[n][d]
		return w

	def gram_mat(self, x):
		gram = [[np.dot(i, j) for i in x] for j in x]
		return gram

	def fit(self, X_train, y_train):
		self.gram = self.gram_mat(X_train)
		is_wrong = False
		iter_n = 0
		while not is_wrong:
			#iter_n += 1
			#print(iter_n)
			wrong_count = 0
			for d in range(len(X_train)):
				X = X_train[d]
				y = y_train[d]
				if y * (self.sign(X_train, y_train, self.alpha, d) + self.b) <= 0:
					self.alpha[d] += self.ita
					self.b += y
					wrong_count += 1
			if wrong_count == 0:
					is_wrong = True
		for j in range(len(X_train)):
			self.w = np.add(self.alpha[j] * X_train[j] * y_train[j], self.w)
		return 'dual form train finish', self.alpha, self.w, self.b

#perceptron = Model()
perceptron = dualModel()
perceptron.fit(X, y)

x_points = np.linspace(4, 7, 10)

y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]

print("w,b=", perceptron.w, perceptron.b)

plt.plot(x_points, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()




'''
### scikit_learn Perceptron
from sklearn.linear_model import Perceptron
clf = Perceptron(fit_intercept=False, n_iter=1000, shuffle=False)
clf.fit(X,y)
print(clf.coef_) # w
print(clf.intercept_) # b
x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0] * x_points + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_points, y_)
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
'''


