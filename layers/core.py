from layer import Layer

from activations import get_activation_functions

import numpy as np


class Dense(Layer):
	def __init__(self, num_hidden, name = '', activation = 'linear'):
		Layer.__init__(self, name)
		self.num_hidden = num_hidden
		self.activation_forwards, self.activation_backwards = get_activation_functions(activation)