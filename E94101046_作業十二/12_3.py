import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

r0, r1 = 0.5, 1.0
theta0, theta1 = 0, np.pi/3
h = 0.1           # Δr
k = np.pi/30      # Δθ
r = np.arange(r0, r1+h, h)
theta = np.arange(theta0, theta1+k, k)
nr = len(r)
nt = len(theta)

T = np.zeros((nr, nt))
T[0, :] = 50          # r = 0.5
T[-1, :] = 100        # r = 1
T[:, 0] = 0           # θ = 0
T[:, -1] = 0          # θ = π/3

def idx(i, j):
    return (i-1)*(nt-2) + (j-1)

N = (nr-2)*(nt-2)  
A = np.zeros((N, N))
b = np.zeros(N)

for i in range(1, nr-1):
    for j in range(1, nt-1):
        row = idx(i, j)
        ri = r[i]
        # Five-point stencil
        # 中心
        A[row, row] = -2/h**2 - 2/(ri**2 * k**2)
        # r+1
        if i+1 < nr-1:
            A[row, idx(i+1, j)] = 1/h**2 + 1/(2*h*ri)
        else:
            b[row] -= (1/h**2 + 1/(2*h*ri)) * T[i+1, j]
        # r-1
        if i-1 > 0-1:
            A[row, idx(i-1, j)] = 1/h**2 - 1/(2*h*ri)
        else:
            b[row] -= (1/h**2 - 1/(2*h*ri)) * T[i-1, j]
        # θ+1
        if j+1 < nt-1:
            A[row, idx(i, j+1)] = 1/(ri**2 * k**2)
        else:
            b[row] -= 1/(ri**2 * k**2) * T[i, j+1]
        # θ-1
        if j-1 > 0-1:
            A[row, idx(i, j-1)] = 1/(ri**2 * k**2)
        else:
            b[row] -= 1/(ri**2 * k**2) * T[i, j-1]

U = np.linalg.solve(A, b)

for i in range(1, nr-1):
    for j in range(1, nt-1):
        T[i, j] = U[idx(i, j)]

R, TH = np.meshgrid(r, theta, indexing='ij')
X = R * np.cos(TH)
Y = R * np.sin(TH)



df = pd.DataFrame(T, index=r, columns=theta)
print(df)