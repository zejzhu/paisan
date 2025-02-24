import numpy as np

a = np.array([1, 0, 1, 0])

shape = a.shape

print(shape[0])

b = np.where(a == 1)

print(b)

c = np.where(a == 1)[0]

print(c)

d = c.shape
print(d[0])