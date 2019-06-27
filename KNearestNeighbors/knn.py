import math
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


def L(x, y, p=2):
	if len(x) == len(y) and len(x) > 1:
		sum = 0
		for i in range(len(x)):
			sum += math.pow(abs(x[i] - y[i]), p)
		return math.pow(sum, 1/p)
	else:
		return 0


x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]
for i in range(1, 5):
	r = {'1-{}'.format(c) : L(x1, c, p=i) for c in [x2, x3]}
	# print(r)
	# print(min(zip(r.values(), r.keys())), 'p =', i)

# def train_test_split(X, y, test_size=0.2):
# 	train_n = len(data) - len(data) * test_size
# 	X_train = X[:train_n+0,:]
# 	X_test = X[train_n:,:]
# 	y_train = y[:train_n+0, :]
# 	y_test = y[train_n:,:]
# 	return X_train, X_test, y_train, y_test


class KNN:
	def __init__(self, X_train, y_train, n_neighbors=3, p=2):
		"""
		parameter: n_neighbors 临近点个数
		parameter: p 距离度量
		"""
		self.n = n_neighbors
		self.p = p
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X):
	    # get n points
		"""
		先把n个点放入列表中，然后遍历训练集所有点，若到测试点的距离小于列表中最大值，则替换，知道保留测试点到训练点集最小的n个点。
	    :param X: 待预测的x点
	    :return: 预测的结果
	    """
		knn_list = []
		for i in range(self.n):
			dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
			knn_list.append((dist, self.y_train[i]))

		for i in range(self.n, len(self.X_train)):
			max_index = knn_list.index(max(knn_list, key=lambda x:x[0]))
			dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
			if knn_list[max_index][0] > dist:
				knn_list[max_index] = (dist, self.y_train[i])

		# stastics
		print(knn_list, '--------------')
		knn = [k[-1] for k in knn_list] # [(0.1414213562373093, -1), (0.1414213562373093, -1), (0.31622776601683766, -1)]
		count_pairs = Counter(knn) # Counter({-1: 3})
		print(count_pairs, '------------------')
		#max_count = sorted(count_pairs, key=lambda x:x)[-1] # -1
		max_count = sorted(count_pairs.items(), key=lambda x:x[1])[-1][0] # -1 be changed by thomas on 12/19/18
		print(max_count, '-----------')
		return max_count

	def score(self, X_test, y_test):
		right_count = 0
		n = 10
		for X, y in zip(X_test, y_test):
			label = self.predict(X)
			if label == y:
				right_count += 1
		return right_count / len(X_test)


if __name__ == "__main__":
	iris = load_iris()
	df = pd.DataFrame(iris.data, columns=iris.feature_names)
	df['label'] = iris.target
	df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
	"""
	### draw raw data
	print(df.label.value_counts())
	plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
	plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
	plt.xlabel('sepal length')
	plt.ylabel('sepal width')
	plt.legend()
	plt.show()
	"""
	data = np.array(df.iloc[:100, [0, 1, -1]])
	X, y = data[:, :-1], data[:, -1]
	y = np.array([1 if i == 1 else -1 for i in y])
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	clf = KNN(X_train, y_train, n_neighbors=10)

	# print(clf.score(X_test, y_test))
	# test_point = [6.0, 3.0]
	# test_point = [4.5, 3.5]
	# test_point = [5.4, 3.0]
	test_point = [5.3, 3.1]
	# a = clf.predict(test_point)
	print("显示单点测试：", test_point)
	print('Test Point: {}'.format(clf.predict(test_point)))
	plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='-1')
	plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
	plt.plot(test_point[0], test_point[1], 'ro', label='test_point')
	plt.xlabel('sepal length')
	plt.ylabel('sepal width')
	plt.legend()
	plt.show()

	print("显示测试集测试：")
	print("test sets' scores:", clf.score(X_test, y_test))

"""
### scikitlearn
print("----------------This is scikilearn result.--------------------")
clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)
print(clf_sk.score(X_test, y_test))
test_point = [[4.5, 3.5], [5.5, 3.5], [6.5, 3.0]]
#print(clf_sk.predict(test_point))
print(test_point, ': {}'.format(clf_sk.predict(test_point)))
"""
