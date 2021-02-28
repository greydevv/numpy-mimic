def shape(array):
	shp = (len(array),)
	if all(isinstance(e, list) for e in array):
		sub_shp = [shape(s) for s in array]
		if sub_shp.count(sub_shp[0]) == len(sub_shp):
			shp += sub_shp[0]
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
	else:
		return [uniform(shape[1:], value)]*shape[0]

def chunks(array, step):
	return [array[i:i+step] for i in range(0, len(array), step)]

def reshape(array, shape):
	if len(shape) == 1:
		return array

	step = int(len(array)/shape[0])
	chunked = [array[i:i+step] for i in range(0, len(array), step)]

	return [reshape(chunk, shape[1:]) for chunk in chunked]









