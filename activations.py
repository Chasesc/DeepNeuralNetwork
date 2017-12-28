import numpy as np

def relu_forwards(Z):
	return np.maximum(0, Z)

def relu_backwards(dA, Z):
	dZ = np.array(dA, copy = True)
	dZ[Z <= 0] = 0

	return dZ

def sigmoid_forwards(Z):
	return 1 / (1 + np.exp(-Z))

def sigmoid_backwards(dA, Z):
	forwards = sigmoid_forwards(Z)
	return dA * forwards * (1 - forwards)

def get_activation_functions(activation):
	import sys
	import inspect

	activation = activation.lower()

	all_functions = dict(inspect.getmembers(sys.modules[__name__], inspect.isfunction))

	forwards_name  = '{activation}_forwards'.format(activation = activation)
	backwards_name = '{activation}_backwards'.format(activation = activation)

	forwards_function = all_functions.get(forwards_name)
	backwards_function = all_functions.get(backwards_name)

	if not forwards_function or not backwards_function:
		raise ValueError('{activation} is an invalid activation function name. (No implementation exists)'.format(activation = activation))

	return forwards_function, backwards_function