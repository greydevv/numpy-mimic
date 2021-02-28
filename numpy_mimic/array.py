"""
TODO
====

Attributes:
	T : ndarray
		The transposed array.
	data: buffer
		Python buffer object pointing to the start of the array's data.
	dtype : dtype object
		Data-type of the array's elements.
	flags: dict
		Information about the memory layout of the array.
	flat : numpy.flatiter object
		A 1-D iterator over the array.
	imag : ndarray
		The imaginary part of the array.
	real : ndarray
		The real part of the array.
	[DONE] size : int
		Number of elements in the array. 
	itemsize : int
		Length of one array element in bytes.
	nbytes : int
		Total bytes consumed by the elements of the array.
	[DONE] ndim : tulpe of ints
		Number of array dimensions.
	[DONE] shape : tuple of ints
		Tuple of array dimensions.
	strides : tuple of ints
		Tuple of bytes to step in each dimension when traversing an array.
	[MAYBE] ctypes : ctypes object
		An object to simplify the interaction of the array with the ctypes module.
	base : ndarray
		Base object if memory is from some other object.

Methods:
	all([axis, out, keepdims, where]) : Returns True if all elements evaluate to True.
	any([axis, out, keepdims, where]) : Returns True if an of the elements of a evaluate to True.
	argmax([axis, out]) : Return indices of the maximum values along the given axis.
	argmin([axis, out]) : Return indices of the minimum values along the given axis.
	argpartition(kth[, axis, kind, order]) : Returns the indices that would partition this array.
	argsort([axis, kind, order]) : Returns the indices that would sort this array.
	astype(dtype[, order, casting, subok, copy]) : Copy of the array, cast to a specified type.
	byteswap([inplace]) : Swap the bytes of the array elements.
	choose(choices[, out, mode]) : Use an index array to construct a new array from a set of choices.
	clip([min, max, out]) : Return an array whose values are limited to [min, max].
	compress(condition[, axis, out]) : Return selected slices of this array along given axis.
	conj() : Complex-conjugate all elements.
	conjugate() : Return the complex conjugate, element-wise.
	copy([order]) : Return a copy of the array
	cumprod([axis, dtype, out]) : Return the cumulative product of the elements along the given axis.
	cumsum([axis, dtype, out]) : return the cumulative sum of the elements along the given axis.
	diagonal([offset, axis1, axis2]) : Return specified diagonals.
	dot(b[, out]) : Dot product of two arrays.
	dump(file) : Dump a pickle of the array to the specified file.
	dumps() : Returns the pickle of the array as a string.
	fill(value) : Fill the array with a scalar value.
	[DONE] flatten([order]) : Return a copy of the array collapsed into one dimension.
	getfield(dtype[, offset]) : Returns a field of the given array as a certain type.
	item(*args) : Copy an element of an array to a standard Python scalar and return it.
	itemset(*args) : Insert scalar into array (scalar is cast to array's dtype, if possible).
	max([axis, out, keepdims, initial, where]) : Return the maximum along a given axis.
	mean([axis, out, keepdims, initial, where]) : Returns the average of the array elements along given axis.
	min([axis, out, keepdims, initial, where]) : Return the minimum along a given axis.
	newbyteorder([new_order]) : Return the array with the same data viewed with a different byte order.
	nonzero() : Return the indices of the elements that are non-zero.
	partition(kth[, axis, kind, order]) : Rearranges the elements in the array in such a way that the value of the element in kth position is in the position it would be in a sorted array.
	prod([axis, dtype, out, keepdims, initial, ...]) : Return the product of the array elements over the given axis.
	ptp([axis, out, keepdims]) : Peak to peak (maximum - minimum) value along a given axis.
	put(indices, values[, mode]) : Set a.flat[n] = values[n] for all n in indices.
	ravel([order]) : Return a flattened array.
	repeat(repeats[, axis]) : Repeat elements of an array.
	[DONE] reshape(shape[, order]) : Returns an array containing the same data with a new shape.
	resize(new_shape[, refcheck]) : Change shape and size of array in-place.
	round([decimals, out]) : Return a with each element rounded to the given number of decimals.
	searchsorted(v[, side, sorter]) : Find indices where elements of v should be inserted in a to maintain order.
	setfield(val, dtype[, offset]) : Put a value into a specified place in a field defined by a data-type.
	setflags([write, align, uic]) : Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY), respectively.
	std([axis, dtype, out, ddof, keepdims, where]) : Returns the standard deviation of the array elements along given axis.
	sum([axis, dtype, out, ddof, keepdims, where]) : Return the sum of the array elements over the given aixs.
	swapaxes(axis1, axis2) : Return the sum of the array elements over the given axis.
	take(indices[, axis, out, mode]) : Return an array formed from the elements of a at the given indices.
	tobytes([order]) : Construct Python bytes containing the raw data bytes in the array.
	tofile(fid[, sep, format])
	[DONE] tolist() : Return the array as an a.ndim-levels deep nested list of Python scalars.
	tostring([order]) : A compatability alias for 'tobytes', with exactly the same behavior.
	trace([offset, axis1, axis2, dtype, out]) : Return the sum along diagonals of the array.
	transpose(*axes) : Returns a view of the array with axes transposed.
	var([axis, dtype, out, ddof, keepdims, where]) : Returns the variance of the array elements, along given axis.
	view([dtype][, type]) : New view of array with the same data.
	__getitem__ : support tuple slicing
"""

from collections import UserList
from functools import reduce
import operator
from numpy_mimic.util._array import reshape as _reshape
from abc import ABC

class Flatiter(object):
	"""
	NumPy does not allow subclassing or instantiation of the 'numpy.flatiter' object.
	If 'numpy.flatiter' is subclassed:
		TypeError: type 'numpy.flatiter' is not an acceptable base type
	If 'numpy.flatiter' is instantiated:
		TypeError: cannot create 'numpy.flatiter' instances

	Therefore, passing an object that is not of type 'numpy_mimc.Array' will not work,
	as the 'Flatiter.__init__' method calls the 'Array.flatten' and 'Array.tolist'
	methods. 
	"""
	def __init__(self, base):
		self.base = base
		self.__data = base.flatten().tolist()
		self.index = 0

	@property
	def coords(self):
		raise NotImplementedError

	def __getitem__(self, i):
		# if isinstance(i, slice):
		# 	return self.__data[i]
		# else:
		# 	return self.__data[i]
		return self.__data[i]
	
	def __iter__(self):
		return self

	def __next__(self):
		try:
			e = self.__data[self.index]
			self.index += 1
			return e
		except IndexError:
			raise StopIteration

class Array(UserList):
	# accepts tuple and converts to list
	# accepts dict and set
	def __init__(self, initlist=[]):
		self.data = []
		if initlist is not None and isinstance(initlist, list):
			self.data = [Array(e) if isinstance(e, list) else e for e in initlist]

	@property
	def size(self):
		"""
		https://numpy.org/doc/stable/reference/generated/numpy.ndarray.size.html#numpy.ndarray.size
		"""
		return len(self.flatten())

	@property
	def ndim(self):
		return len(self.shape)

	@property
	def flat(self):
		return Flatiter(self)

	@property
	def shape(self):
		"""
		https://numpy.org/doc/stable/reference/generated/numpy.shape.html#numpy.shape
		"""

		# TODO: think about calculating shape in 'Array.__init__'
		shp = (len(self.data),)
		if all(isinstance(e, self.__class__) for e in self.data):
			# becomes recursive through children with 's.shape' (below)
			sub_shp = [s.shape for s in self.data]
			if sub_shp.count(sub_shp[0]) == len(sub_shp):
				shp += sub_shp[0]
		return shp

	def flatten(self):
		"""
		https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html#numpy.ndarray.flatten
		"""

		# TODO: add 'order' parameter
		result = []
		for e in self.data:
			if isinstance(e, self.__class__):
				# becomes recursive through children with 'e.flatten()' (below)
				result.extend(e.flatten())
			else:
				result.append(e)

		return self.__class__(result)

	def reshape(self, shape):
		"""
		https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape
		"""

		# TODO: think about replacing this with class recursion via children
		# TODO: add 'order' parameter
		_size = self.size
		if reduce(operator.mul, shape, 1) != _size:
			raise ValueError(f"cannot reshape {self.__class__.__name__} of size {_size} into shape {shape}")

		result = _reshape(self.flatten(), shape)
		return self.__class__(result)

	def tolist(self):
		return [list(e.tolist()) if isinstance(e, self.__class__) else e for e in self.data]

	def __getitem__(self, i):
		if isinstance(i, slice):
			return self.__class__(self.data[i])
		elif isinstance(i, tuple):
			raise NotImplementedError
		elif isinstance(i, bool):
			raise NotImplementedError
		else:
			return self.data[i]

	def __gt__(self, other):
		return all(e > other for e in self.flat)

	def __repr__(self):
		# TODO: format output, see 'Array.__str__' for more info
		return f"{self.__class__.__name__}({self.data})"

	def __str__(self):
		# TODO: format output
		# each line has 72 characters, evenly spaced to line up down the array
		# [     90      91      92      93      94      95      96      97      98
		#      99     100     101     102     103     104     105     106 4299890]
		return str(self.data)
























