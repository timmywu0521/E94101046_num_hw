import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

K = 0.1
dr = 0.1
dt = 0.5
r0, r1 = 0.5, 1.0
t_end = 10

r = np.arange(r0, r1 + dr, dr)
N = len(r)
t = np.arange(0, t_end + dt, dt)
M = len(t)

T = np.zeros((M, N))
T[0, :] = 200 * (r - 0.5)

lam = 4 * K * dt / dr**2

for n in range(0, M-1):
    # 右邊界 
    T[n+1, -1] = 100 + 40 * t[n+1]

    # 內部節點
    for j in range(1, N-1):
        rj = r[j]
        T[n+1, j] = (
            T[n, j]
            + 4 * K * dt * (
                (T[n, j+1] - 2*T[n, j] + T[n, j-1])/dr**2
                + (T[n, j+1] - T[n, j-1])/(2*dr*rj)
            )
        )

    # 左邊界 
    # (T[1] - T[0])/dr + 3*T[0] = 0  ==> T[0] = T[1] / (1 + 3*dr)
    T[n+1, 0] = T[n+1, 1] / (1 + 3*dr)



df = pd.DataFrame(T, index=t, columns=r)
print(df)