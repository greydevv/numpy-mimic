"""
TODO
====

Attributes:
    T : ndarray
        The transposed array.
    [MAYBE] data: buffer
        Python buffer object pointing to the start of the array's data.
    dtype : dtype object
        Data-type of the array's elements.
    [MAYBE] flags: dict
        Information about the memory layout of the array.
    [DONE] flat : numpy.flatiter object
        A 1-D iterator over the array.
    [DONE] imag : ndarray
        The imaginary part of the array.
    [DONE] real : ndarray
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

from functools import reduce
import operator
from numpy_mimic.util import reshape as _reshape

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
        if isinstance(i, slice):
            return Array(self.__data[i])
        else:
            return self.__data[i]

    def __len__(self):
        return len(self.__data)
    
    def __iter__(self):
        return self

    def __next__(self):
        try:
            e = self.__data[self.index]
            self.index += 1
            return e
        except IndexError:
            raise StopIteration

class Array(object):
    # accepts tuple and converts to list
    # accepts dict and set
    # check for ragged lists, raise error (or warning?)
    def __init__(self, obj=[]):
        if obj is not None and isinstance(obj, list):
            if isinstance(obj, list):
                self.data = [Array(e) if isinstance(e, list) else e for e in obj]
            else:
                pass
                # try:
                #   obj = list(obj)
                # except 

    @property
    def size(self):
        """
        https://numpy.org/doc/stable/reference/generated/numpy.ndarray.size.html#numpy.ndarray.size
        """
        # TODO: think about using math to get this value from the shape?
        # Multiply all elements in 'self.shape' to get the size of the 'Array'.
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
            sub_shp = [s.shape for s in self.data]
            if sub_shp and sub_shp.count(sub_shp[0]) == len(sub_shp):
                shp += sub_shp[0]
        return shp

    @property
    def imag(self):
        result = []
        for e in self.data:
            if isinstance(e, (self.__class__, int, float, complex)):
                result.append(e.imag)
            else:
                # TODO: what if value is a 'str'?
                result.append(e)

        return self.__class__(result)

    @property
    def real(self):
        result = []
        for e in self.data:
            if isinstance(e, (self.__class__, int, float, complex)):
                result.append(e.real)
            else:
                # TODO: what if value is a 'str'?
                result.append(e)

        return self.__class__(result)

    def all(self):
        # TODO: implement 'axis' parameter and others
        return all(e for e in self.flatten())

    def take(self, indices, axis=None):
        if axis is None:
            flattened = self.flatten()[indices]
            return [flattened[i] for i in indices]

    def fill(self, value):
        # TODO: I don't like this two-method implementation
        self.data = self.__fill(value)

    def __fill(self, value):
        # TODO: I don't like this two-method implementation
        result = []
        for e in self.data:
            if isinstance(e, self.__class__):
                result.append(e.__fill(value))
            else:
                result.append(value)

        return self.__class__(result)

    def transpose(self):
        """
        Base case: 2D array, easy transposition using 'zip' method.

        1. Flatten the data to a 2D array and zip them together.
        2. Based on shape, compose list of indices using 'range'
        3. Select from 2D zipped array using list of indices
            a. [zipped[i] for i in indices]
        """

        # can self.ndim be zero?
        if self.ndim == 2:
            result = list(map(list, zip(*self.data)))
            return self.__class__(result)
        else:
            return self if self.ndim <= 1 else self._transpose()


    def _transpose(self):
        # result = self.__class__(list(map(list, zip(*[e.flatten() for e in self.data])))).flatten().reshape((self.shape[::-1]))
        # result = [list(map(list, zip(*map(Array.flatten, self.data))))]
        # return result
        new_shp = self.shape[::-1]
        result = []
        if self.ndim == 1:
            pass

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

    def reshape(self, *shape):
        """
        https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape
        """
        print(shape)
        # TODO: think about replacing this with class recursion via children
        # TODO: add 'order' parameter
        if len(shape) == 1 and isinstance(shape[0], tuple):
            # supports receiving 'shape' as tuple or varargs
            shape = shape[0]

        if reduce(operator.mul, shape, 1) != self.size:
            raise ValueError(f"cannot reshape {self.__class__.__name__} of size {self.size} into shape {shape}")

        result = _reshape(self.flatten().data, shape)
        return self.__class__(result)

    def tolist(self):
        return [list(e.tolist()) if isinstance(e, self.__class__) else e for e in self.data]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.__class__(self.data[i])
        elif isinstance(i, tuple):
            if len([e for e in i if e is not None]) > self.ndim:
                raise IndexError(f"too many indices for array: array is {self.ndim}-dimensional, but {len(i)} were indexed.")
    
            # result = self.__slice(i)
            result = self.aindex(i)
            return self.__class__(result)
        elif isinstance(i, list):
            raise NotImplementedError
        elif isinstance(i, bool):
            raise NotImplementedError
        else:
            # TODO: raise IndexError here, get 'axis' somehow
            return self.data[i]

    def aindex(self, index):
        pass

    # def __slice(self, i, axis=0):
    #   try:
    #       index = i[0]
    #       print(f"INDEX: {index}")
    #   except IndexError as e:
    #       print("ERROR")
    #       return
    #   if isinstance(index, slice):
    #       result = []
    #       for e in self.data[index]:
    #           if isinstance(e, self.__class__):
    #               if len(i) > 1:
    #                   result.append(e.__slice(i[1:], axis+1))
    #           else:
    #               result.append(e)
    #       return result
    #       # return [e._slice(i[1:], axis+1) if isinstance(e, self.__class__) else e for e in self.data[index]]
    #   elif isinstance(index, type(None)):
    #       # Increases dimension of output by 1 dimension.
    #       # 'numpy.newaxis' is used interchangeably with 'None' in the NumPy library.
    #       # return [self.__slice(i[1:])]
    #       return [self[i[1:]]]
    #   else:
    #       if index > len(self)-1:
    #           raise IndexError(f"index {index} is out of bounds for axis {axis} with size {len(self.data)}")
    #       e = self.data[index]
    #       if isinstance(e, self.__class__):
    #           return e.__slice(i[1:], axis+1)
    #       else:
    #           return e


    def __slice(self, i):
        index = i[0]
        nxt = None if len(i) == 1 else i[1:]
        if isinstance(index, slice):
            if nxt:
                return [e.__slice(nxt) if isinstance(e, self.__class__) else e for e in self.data[index]]
            else:
                return self.data[index]
        
        elif isinstance(index, type(None)):
            return [self.__slice(i[1:])]

        else:
            e = self.data[index]
            if isinstance(e, self.__class__):
                return e.__slice(i[1:])
            else:
                return e


    
    def __gt__(self, other):
        return all(e > other for e in self.flat)

    def __ge__(self, other):
        return all(e >= other for e in self.flat)

    def __lt__(self, other):
        return all(e < other for e in self.flat)

    def __le__(self, other):
        return all(e <= other for e in self.flat)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        # TODO: format output
        # each line has 72 characters, evenly spaced to line up down the array
        # [     90      91      92      93      94      95      96      97      98
        #       99     100     101     102     103     104     105     106 4299890]
        return str(self.data)

    __repr__ = __str__



























