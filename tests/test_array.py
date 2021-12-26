import pytest
from numpy_mimic import Array
import numpy as np

"""
Test Cases
----------

base = np.arange(6).reshape(2,3)
base = np.arange(12).reshape(4,3,1)
base = np.arange(280).reshape(4,7,5,2)
base = np.arange(960).reshape(2,3,10,2,4,2)
base = np.arange(1).reshape(1, 1, 1, 1, 1, 1, 1, 1, 1)
base = np.arange(64).reshape(1, 1, 2, 1, 1, 4, 1, 8, 1)
"""

def test_size():
	base = np.arange(6).reshape(2,3)
	assert Array(base.tolist()).size == base.size
	base = np.arange(12).reshape(4,3,1)
	assert Array(base.tolist()).size == base.size
	base = np.arange(280).reshape(4,7,5,2)
	assert Array(base.tolist()).size == base.size
	base = np.arange(960).reshape(2,3,10,2,4,2)
	assert Array(base.tolist()).size == base.size
	base = np.arange(1).reshape(1,1,1,1,1,1,1,1,1)
	assert Array(base.tolist()).size == base.size
	base = np.arange(64).reshape(1,1,2,1,1,4,1,8,1)
	assert Array(base.tolist()).size == base.size

def test_ndim():
	base = np.arange(6).reshape(2,3)
	assert Array(base.tolist()).ndim == base.ndim
	base = np.arange(12).reshape(4,3,1)
	assert Array(base.tolist()).ndim == base.ndim
	base = np.arange(280).reshape(4,7,5,2)
	assert Array(base.tolist()).ndim == base.ndim
	base = np.arange(960).reshape(2,3,10,2,4,2)
	assert Array(base.tolist()).ndim == base.ndim
	base = np.arange(1).reshape(1,1,1,1,1,1,1,1,1)
	assert Array(base.tolist()).ndim == base.ndim
	base = np.arange(64).reshape(1,1,2,1,1,4,1,8,1)
	assert Array(base.tolist()).ndim == base.ndim

def test_shape():
	base = np.arange(6).reshape(2,3)
	assert Array(base.tolist()).shape == base.shape
	base = np.arange(12).reshape(4,3,1)
	assert Array(base.tolist()).shape == base.shape
	base = np.arange(280).reshape(4,7,5,2)
	assert Array(base.tolist()).shape == base.shape
	base = np.arange(960).reshape(2,3,10,2,4,2)
	assert Array(base.tolist()).shape == base.shape
	base = np.arange(1).reshape(1,1,1,1,1,1,1,1,1)
	assert Array(base.tolist()).shape == base.shape
	base = np.arange(64).reshape(1,1,2,1,1,4,1,8,1)
	assert Array(base.tolist()).shape == base.shape

def test_flatten():
	base = np.arange(6).reshape(2,3)
	assert Array(base.tolist()).flatten().tolist() == base.flatten().tolist()
	base = np.arange(12).reshape(4,3,1)
	assert Array(base.tolist()).flatten().tolist() == base.flatten().tolist()
	base = np.arange(280).reshape(4,7,5,2)
	assert Array(base.tolist()).flatten().tolist() == base.flatten().tolist()
	base = np.arange(960).reshape(2,3,10,2,4,2)
	assert Array(base.tolist()).flatten().tolist() == base.flatten().tolist()
	base = np.arange(1).reshape(1,1,1,1,1,1,1,1,1)
	assert Array(base.tolist()).flatten().tolist() == base.flatten().tolist()
	base = np.arange(64).reshape(1,1,2,1,1,4,1,8,1)
	assert Array(base.tolist()).flatten().tolist() == base.flatten().tolist()

def test_reshape():
	base = np.arange(6).reshape(2,3)
	assert Array(base.tolist()).reshape((1,1,3,2)).tolist() == base.reshape(1,1,3,2).tolist()
	base = np.arange(12).reshape(4,3,1)
	assert Array(base.tolist()).reshape((3,2,2)).tolist() == base.reshape(3,2,2).tolist()
	base = np.arange(280).reshape(4,7,5,2)
	assert Array(base.tolist()).reshape((2,2,5,2,7)).tolist() == base.reshape(2,2,5,2,7).tolist()
	base = np.arange(960).reshape(2,3,10,2,4,2)
	assert Array(base.tolist()).reshape((2,5,2,3,4,4)).tolist() == base.reshape(2,5,2,3,4,4).tolist()
	base = np.arange(1).reshape(1,1,1,1,1,1,1,1,1)
	assert Array(base.tolist()).reshape((1,)).tolist() == base.reshape(1).tolist() # <---- [FAILURE] got [], expected [0]
	base = np.arange(64).reshape(1,1,2,1,1,4,1,8,1)
	assert Array(base.tolist()).reshape((1,2,4,8)).tolist() == base.reshape(1,2,4,8).tolist()











