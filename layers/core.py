from layer import Layer

import numpy as np

class Dense(Layer):
	def __init__(self, num_hidden, name = '', activation = 'linear'):
		Layer.__init__(self, name)
		self.num_hidden = num_hidden
		self.activation = activation

	def activation_function(self, x):
		return np.maximum(0, x)


def main():
	dense = Dense(5, name = 'dense1')

	print(dense._W)
	print(dense.name)

if __name__ == '__main__':
	main()