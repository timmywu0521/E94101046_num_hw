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

alpha = 4 * K * dt / (2 * dr**2)

for n in range(0, M-1):
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)
    T_next = np.zeros(N)
    T_next[-1] = 100 + 40 * t[n+1]

    # 內部點
    for j in range(1, N-1):
        rj = r[j]
        beta = 4 * K * dt / (4 * dr * rj)
        a[j] = -alpha + beta
        b[j] = 1 + 2 * alpha
        c[j] = -alpha - beta
        # 右
        lap_Tn = (T[n, j+1] - 2*T[n, j] + T[n, j-1]) / dr**2
        der_Tn = (T[n, j+1] - T[n, j-1]) / (2*dr*rj)
        d[j] = T[n, j] + alpha * (lap_Tn + der_Tn)

    # 左邊界 
    b[0] = 1
    c[0] = -1 / (1 + 3*dr)
    d[0] = 0

    a[-1] = 0
    b[-1] = 1
    c[-1] = 0
    d[-1] = T_next[-1]

    cp = np.zeros(N)
    dp = np.zeros(N)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for j in range(1, N):
        denom = b[j] - a[j] * cp[j-1]
        cp[j] = c[j] / denom if j < N-1 else 0
        dp[j] = (d[j] - a[j] * dp[j-1]) / denom
    T_next[-1] = dp[-1]
    for j in range(N-2, -1, -1):
        T_next[j] = dp[j] - cp[j] * T_next[j+1]
    T[n+1, :] = T_next



df = pd.DataFrame(T, index=t, columns=r)
print(df)