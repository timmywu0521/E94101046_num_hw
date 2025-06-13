import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dx = dt = 0.1
x = np.arange(0, 1 + dx, dx)
t = np.arange(0, 1 + dt, dt)  
Nx = len(x)
Nt = len(t)

p = np.zeros((Nt, Nx))
p[0, :] = np.cos(2 * np.pi * x)
p[0, 0] = 1
p[0, -1] = 2

for i in range(1, Nx - 1):
    p[1, i] = (p[0, i + 1] + p[0, i - 1]) / 2 \
              + dt * 2 * np.pi * np.sin(2 * np.pi * x[i]) \
              + (1 - dt**2) * p[0, i] / 2
p[1, 0] = 1
p[1, -1] = 2

for n in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        p[n + 1, i] = (dt ** 2) * (p[n, i + 1] - 2 * p[n, i] + p[n, i - 1]) / (dx ** 2) \
                      + 2 * p[n, i] - p[n - 1, i]
    p[n + 1, 0] = 1
    p[n + 1, -1] = 2



df = pd.DataFrame(p, index=t, columns=x)
print(df)