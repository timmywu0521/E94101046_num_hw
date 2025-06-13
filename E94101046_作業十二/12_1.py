import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pi = np.pi
h = k = 0.1 * pi
x = np.arange(0, pi + h, h)
y = np.arange(0, pi/2 + k, k)
nx, ny = len(x), len(y)

u = np.zeros((nx, ny))
u[0, :] = np.cos(y)         # u(0, y)
u[-1, :] = -np.cos(y)       # u(pi, y)
u[:, 0] = np.cos(x)         # u(x, 0)
u[:, -1] = 0                # u(x, pi/2)

N = (nx-2)*(ny-2)
A = np.zeros((N, N))
b = np.zeros(N)

def idx(i, j):
    # map 2D (i, j) to 1D index in Ax=b
    return (i-1)*(ny-2) + (j-1)

for i in range(1, nx-1):
    for j in range(1, ny-1):
        row = idx(i, j)
        A[row, row] = -4
        # 上
        if j+1 < ny-1:
            A[row, idx(i, j+1)] = 1
        else:
            b[row] -= u[i, j+1]
        # 下
        if j-1 > 0-1:
            A[row, idx(i, j-1)] = 1
        else:
            b[row] -= u[i, j-1]
        # 右
        if i+1 < nx-1:
            A[row, idx(i+1, j)] = 1
        else:
            b[row] -= u[i+1, j]
        # 左
        if i-1 > 0-1:
            A[row, idx(i-1, j)] = 1
        else:
            b[row] -= u[i-1, j]

U = np.linalg.solve(A, b)

for i in range(1, nx-1):
    for j in range(1, ny-1):
        u[i, j] = U[idx(i, j)]

X, Y = np.meshgrid(x, y, indexing='ij')

df = pd.DataFrame(u, index=x, columns=y)
print(df)