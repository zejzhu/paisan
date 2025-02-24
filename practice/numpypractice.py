import numpy as np

#numpy arrays and matrices (2d+ arrays) must be the same data type
a = np.array([2,4,5,2,8,2523452])

#wow a matrix
a = np.array([[4,235235,2],[324,3,9]])

#an array w 3 zeros (floats)
a = np.zeros(3)

#a 2x4 matrix (2 rows 4 columns) (also floats)
a = np.ones((2,4))
#print(a)

"""
array attributes:
ndim (# of array dimensions like 3 dimensional array)
shape (tuple of array dimensions like 2 by 4 by 3)
size (number of elements in array)
dtype (type of item)
"""

rng = np.random.default_rng(334) #334 is a seed for random number generation

#array of 3 random floats
x1 = rng.random(4)
#print x3

#5x4 matrix of integers between 0 and 20
x2 = rng.integers(0, 20, size=(5,4))
print(x2)

#5x4x3 (or 5 4x3 matrices) of random floats
x3 = rng.random((5,4,3))
#print(x3)

#5x4 matrix of integers between 0 and 10
x4 = rng.integers(0, 10, size=(5,4))
#print(x4)

"""
array indexing

indexing a specific element: a[5]

indexing a range of elements: a[start:stop:step]
(unspecified values default to start = 0, stop = size of dimension, step = 1)

indexing matrix dimensions is separated by a comma: a[5, 4]
"""

#from 0,0 to 2,endofdimension, with step size = 1
#aka print all items in the first two rows
#print(x2[:2, :])

#print from 0 to endofdimension, 1
#aka print column 1
#print(x2[:, 1])

"""
numpy operations on matrices

we have the classic + - / * %

we can aggregate functions using methods like
np.sum, np.prod (product), np.mean, np.max, np.min,
np.argmax (index of max value), np.argmin,
np.all (all elements are true), np.any (any elements are true)


also matrix operators like
Transpose: a.T
matrix multiplication: a@b or np.matmul(a,b)
determinant: np.linalg.det(a)
inverse of square matrix: np.linalg.inv(a)

and theres many more in numpy ! lots of stuff u can do
"""
