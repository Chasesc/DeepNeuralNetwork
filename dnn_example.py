from dnn import DNN
from layers.core import Dense

import numpy as np
import h5py

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

	# Standardize data to have feature values between 0 and 1.
	train_x = train_x_flatten/255.
	test_x = test_x_flatten/255.

	return train_x, test_x, train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def main():
	options = {
		'learning_rate' : 0.1,
		'beta1' : 0.9,
		'optimizer' : 'gd',
		'loss' : 'crossentropy'
	}

	train_x, test_x, train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()

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