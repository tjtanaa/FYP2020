import torch
from .ARTN import ARTN

class model_factory(object):

	def __init__(self, model_name='ARTN'):
		pass

	def get_model(self):
		return ARTN
		