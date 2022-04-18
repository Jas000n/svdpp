import numpy as np

k = 3
n = 4
m = 5

p = np.zeros((k, m))  # matrix of user preference
q = np.zeros((n, k))  # matrix of item quality
print(q)
print("----------------------")
qi = q[0]
print(qi)
print("----------------------")
print(p)
print("----------------------")
pi = p[:,0]
print(pi)
print("----------------------")
print(qi.dot(pi))
