from layers.core import Dense

import numpy as np

_default_params = {
	'learning_rate' : 0.001,
	'beta1' 		: 0.9,
	'beta2' 		: 0.999,
	'mb_size' 		: 128,
	'num_epochs' 	: 50,
	'optimizer'		: 'adam'
}

# maps from option to a function to check if that option is valid.
_option_restrictions = {
	'learning_rate' : lambda x : x > 0,
	'beta1' 		: lambda x : 0 < x < 1,
	'beta2' 		: lambda x : 0 < x < 1,
	'mb_size' 		: lambda x : x >= 1 and isinstance(x, int),
	'num_epochs' 	: lambda x : x >= 1 and isinstance(x, int),
	'optimizer' 	: lambda x : x in set(['gd', 'gdm', 'adam']),
	'loss'			: lambda x : x in set(['crossentropy'])
}

class DNN(object):
	def __init__(self, X, Y, layers, options, quiet = True):
		self.X = X
		self.Y = Y

		self.options = options
		self._validate_options()

		self.layers = layers
		self.params = self._init_params()

	def train(self):
		pass

	def predict(self):
		self._forward_pass()

		L = len(self.layers)

		return self.cache["A"+str(L)]

	def _forward_pass(self):
		self.cache = {'A0' : self.X}

		for l, layer in enumerate(self.layers):
			self.cache["Z" + str(l+1)] = np.dot(self.params["W"+str(l+1)], self.cache["A"+str(l)]) + self.params["b"+str(l+1)]
			self.cache["A" + str(l+1)] = layer.activation_function(self.cache["Z" + str(l+1)])

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