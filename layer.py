
class Layer(object):
	def __init__(self, name):
		self.name = name
		self._W = None
		self._b = None

	def __repr__(self):
		return 'Layer: {0}'.format(self.name)