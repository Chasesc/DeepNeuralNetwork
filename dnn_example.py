from dnn import DNN
from layers.core import Dense

import numpy as np

def main():
	options = {
		'learning_rate' : 0.1,
		'beta1' : 0.9,
		'optimizer' : 'gd',
		'loss' : 'crossentropy'
	}

	X = np.array([[1, 2], [1, 2], [4, 2]])
	Y = np.array([[0], [0], [0]])

	print(X.shape)
	print(Y.shape)

	layers = [
		Dense(32, activation = 'relu'),
		Dense(5,  activation = 'relu'),
		Dense(1, activation = 'softmax')
	]

	print(len(layers))

	dnn = DNN(X, Y, layers, options)

	for param in sorted(dnn.params):
		print(param, dnn.params[param].shape)

	print(dnn)
	dnn.train()

	print(dnn.predict())

if __name__ == '__main__':
	main()