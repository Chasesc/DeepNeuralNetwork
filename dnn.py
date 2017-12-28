from layers.core import Dense

import losses

import numpy as np

_default_params = {
	'learning_rate'		: 0.001,
	'beta1'			: 0.9,
	'beta2'			: 0.999,
	'mb_size'		: 128,
	'num_epochs'		: 50,
	'optimizer'		: 'adam'
}

# maps from option to a function to check if that option is valid.
_option_restrictions = {
	'learning_rate'		: lambda x : x > 0,
	'beta1'			: lambda x : 0 < x < 1,
	'beta2'			: lambda x : 0 < x < 1,
	'mb_size'		: lambda x : x >= 1 and isinstance(x, int),
	'num_epochs'		: lambda x : x >= 1 and isinstance(x, int),
	'optimizer'		: lambda x : x in set(['gd', 'gdm', 'adam']),
	'loss'			: lambda x : x in set(['crossentropy'])
}

class DNN(object):
	def __init__(self, X, Y, layers, options):
		self.X = X
		self.Y = Y

		self.options = options
		self._validate_options()

		self.layers = layers
		self.params = self._init_params()
		self.loss = losses.get_loss_function(self.options['loss'])

	def train(self, quiet = True):
		self._forward_pass(self.X)
		self._backward_pass()

	def predict(self, X):
		self._forward_pass(X)

		L = len(self.layers)

		return self.cache["A"+str(L)]

	def _forward_pass(self, A0):
		self.cache = {'A0' : A0}

		for l, layer in enumerate(self.layers):
			self.cache["Z" + str(l+1)] = np.dot(self.params["W"+str(l+1)], self.cache["A"+str(l)]) + self.params["b"+str(l+1)]
			self.cache["A" + str(l+1)] = layer.activation_forwards(self.cache["Z" + str(l+1)])

	def _backward_pass(self):
		self.gradients = {}

		L = len(self.layers)
		AL = self.cache["A" + str(L)]

		self.Y = self.Y.reshape(AL.shape)

		dAL = - (np.divide(self.Y, AL) - np.divide(1 - self.Y, 1 - AL))
		#grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
		dZ = self.layers[-1].activation_backwards(dAL, self.cache["Z" + str(L)])
		#dA_prev, dW, db = linear_backward(dZ, linear_cache)
		A_prev = self.cache["A" + str(L - 1)]
		W = self.params["W" + str(L)]
		b = self.params["b" + str(L)]
		m = A_prev.shape[1]
		
		dW = np.dot(dZ, A_prev.T) / m
		db = np.sum(dZ, axis = 1, keepdims = True) / m
		dA_prev = np.dot(W.T, dZ)

		self.gradients["dA" + str(L)], self.gradients["dW" + str(L)], self.gradients["db" + str(L)] = dA_prev, dW, db

		for l in reversed(range(L-1)):
			#linear_activation_backward(grads["dA" + str(l + 2)], (lin-(A, W, b)), actZ), "relu")
			dZ = self.layers[l].activation_backwards(self.gradients["dA" + str(l + 2)], self.cache["Z" + str(l)])

			A_prev = self.cache["A" + str(l - 1)]
			W = self.params["W" + str(l)]
			b = self.params["b" + str(l)]
			m = A_prev.shape[1]

			dW = np.dot(dZ, A_prev.T) / m
			db = np.sum(dZ, axis = 1, keepdims = True) / m
			dA_prev = np.dot(W.T, dZ)

			self.gradients["dA" + str(l + 1)], self.gradients["dW" + str(l + 1)], self.gradients["db" + str(l + 1)] = dA_prev, dW, db

	def _init_params(self):
		params = {}

		layer_dims = [self.X.shape[0], *[layer.num_hidden for layer in self.layers if isinstance(layer, Dense)]]

		for l in range(1, len(layer_dims)):
			params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1]) # Assuming He initialization for now
			params['b' + str(l)] = np.zeros((layer_dims[l], 1)) # Assuming zero initialization for now

		return params

	def _validate_options(self):
		# Set each option to it's default value if one was not given
		not_given_opts = _default_params.keys() - self.options.keys()
		for key in not_given_opts:
			self.options[key] = _default_params[key]

		# Ensure our parameters meet our restrictions
		for opt, evaluate_restriction in _option_restrictions.items():
			if not evaluate_restriction(self.options[opt]):
				raise ValueError("{opt} cannot be {val}. See _option_restrictions in dnn.py.".format(opt=opt, val=self.options[opt]))


	def __repr__(self):
		opt = '\n'.join(str(k) + " : " + str(v) for k, v in self.options.items())
		display = 'DNN with options:\n{opt}'.format(opt=opt)

		return display