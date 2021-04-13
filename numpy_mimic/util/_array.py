def shape(array):
	shp = (len(array),)
	if all(isinstance(e, list) for e in array):
		sub_shp = [shape(s) for s in array]
		if sub_shp:
			if sub_shp.count(sub_shp[0]) == len(sub_shp):
				shp += sub_shp[0]
			else:
				raise ValueError("Ragged data (shapes do not match)!")
	elif any(isinstance(e, list) for e in array):
		raise ValueError("Ragged data (found instance of list and other)!")
	return shp

def flatten(array):
	result = []
	for e in array:
		if isinstance(e, list):
			result.extend(flatten(e))
		else:
			result.append(e)
	return result

def ones(shape, dtype=float):
	if isinstance(shape, int):
		shape == (shape,)
	value = 1.0 if dtype == float else 1
	return uniform(shape, value)

def zeros(shape, dtype=float):
	if isinstance(shape, int):
		shape == (shape,)
	value = 1.0 if dtype == float else 1
	return uniform(shape, value)

def uniform(shape, value=None):
	if len(shape) == 1:
		return [value]*shape[0]
	
	return [uniform(shape[1:], value)]*shape[0]

def chunks(array, step):
	return [array[i:i+step] for i in range(0, len(array), step)]

def reshape(array, shape):
	if len(shape) == 1:
		return array

	step = int(len(array)/shape[0])
	chunked = [array[i:i+step] for i in range(0, len(array), step)]

	return [reshape(chunk, shape[1:]) for chunk in chunked]

def align(s1, s2):
	l1, l2 = len(s1), len(s2)
	if l1 < l2:
		s1 = tuple(1 for _ in range(l2-l1)) + s1
	elif l1 > l2:
		s2 = tuple(1 for _ in range(l1-l2)) + s2
	return s1, s2 

def compatible(s1, s2):
	"""

	Returns 'True' if the two shapes are compatible, else 'False'. The compatibility of the two shapes are determined by the following properties:
	
	1. The arrays all have exactly the same shape.
	2. The arrays all have the same number of dimensions and the length of each dimensions is either a common length or 1.
	3. The arrays that have too few dimensions can have their shapes prepended with a dimension of length 1 to satisfy property 2.

	From (https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting)
	"""
	if len(s1) != len(s2):
		s1,s2 = align(s1,s2)
	return all((x == 1 or y == 1) or x == y for x,y in zip(s1,s2))

def broadcast_shape(s1, s2):
	"""
	Aligns and creates one shape from two which is useful in broadcasting. This method will not throw an error if the two shapes are not compatible, so it is recommended to use this method with the 'compatible' method.
	"""
	s1, s2 = align(s1, s2)
	return tuple(x if y == 1 else y for x,y in zip(s1,s2))












