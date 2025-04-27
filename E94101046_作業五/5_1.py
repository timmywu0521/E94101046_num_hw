import numpy as np

# 定義微分方程
def f(t, y):
    return 1 + (y / t) + (y / t)**2

# 真實解
def exact_solution(t):
    return t * np.tan(np.log(t))

# 計算 f_t
def f_t(t, y):
    fy = (1 / t) + (2 * y / t**2)
    ft = -(y / t**2) - (2 * y**2 / t**3)
    return ft + fy * f(t, y)

# 初始條件
t0 = 1
y0 = 0
h = 0.1
t_end = 2

# 建立時間點
t_values = np.arange(t0, t_end + h, h)
n = len(t_values)

# Euler方法近似解
y_euler = np.zeros(n)
y_euler[0] = y0

for i in range(1, n):
    y_euler[i] = y_euler[i-1] + h * f(t_values[i-1], y_euler[i-1])

# Taylor 2nd order 方法近似解
y_taylor2 = np.zeros(n)
y_taylor2[0] = y0

for i in range(1, n):
    y_taylor2[i] = (y_taylor2[i-1] 
                    + h * f(t_values[i-1], y_taylor2[i-1]) 
                    + (h**2 / 2) * f_t(t_values[i-1], y_taylor2[i-1]))

# 真實解
y_exact = exact_solution(t_values)

# 顯示結果
print("t\tEuler\t\tExact\t\tError (Euler)")
for i in range(n):
    print(f"{t_values[i]:.1f}\t{y_euler[i]:.6f}\t{y_exact[i]:.6f}\t{abs(y_exact[i] - y_euler[i]):.6f}")

print("\n---\n")

print("t\tTaylor 2nd\tExact\t\tError (Taylor 2nd)")
for i in range(n):
    print(f"{t_values[i]:.1f}\t{y_taylor2[i]:.6f}\t{y_exact[i]:.6f}\t{abs(y_exact[i] - y_taylor2[i]):.6f}")
