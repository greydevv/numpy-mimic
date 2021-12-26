import pytest
from numpy_mimic import Array, Flatiter
import numpy as np


def test_flat():
	base = np.arange(6).reshape(2,3)
	array = Array(base.tolist())
	assert isinstance(array.flat, Flatiter)
	assert len(array.flat) == len(base.flat)
	assert isinstance(array.flat[:], Array)