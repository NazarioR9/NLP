import numpy as np
from typing import List

class Compose:
	def __init__(self, *augs):
		self.augs = augs

	def __call__(self, x: List):
		for aug in augs:
			x = aug(*x)
		return x

class BaseTransform:
	def __init__(self, p=0.5):
		self.p =p

	def __call__(self, x: List, *args):
		if np.random.rand() < self.p:
			x = self.apply(*x, *args)
		return x

class SwitchSourceTarget(BaseTransform):
	"""
		Switch source text with target text.
	"""
	def __init__(self, p=0.3):
		super().__init__(p)

	def apply(self, src_text, trg_text):
		return trg_text, src_text

class TruncateText:
	"""
		Shorten the input text based on pct.

		Example:
			text = "I love living in France"
			pct = 0.8

			output = "I love living" / "love living in France"
	"""
	def __init__(self, p=0.3, pct=0.8):
		super().__init__(p)
		self.pct = 0.8

	def apply(self, text):
		splits = text.split()
		length = len(splits)

		bound, size = (1-self.pct)*length, self.pct*length
		start = np.random.randint(int(bound))
		size = int(size)

		return " ".join(splits[start:start+size])