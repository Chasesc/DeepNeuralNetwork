import numpy as np

def crossentropy(Y_hat, Y, epsilon = 1e-10):
	m = Y.shape[1]
	loss = - np.sum(np.multiply(Y, np.log(Y_hat + epsilon)) + np.multiply(1 - Y, np.log(1 - Y_hat + epsilon))) / m

	return np.squeeze(loss)

def get_loss_function(loss):
	import sys
	import inspect

	loss = loss.lower()

	all_functions = dict(inspect.getmembers(sys.modules[__name__], inspect.isfunction))

	loss_function = all_functions.get(loss)

	if not loss_function:
		raise ValueError('{loss} is an invalid loss function name. (No implementation exists)'.format(loss = loss))

	return loss_function