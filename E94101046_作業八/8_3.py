import numpy as np
from scipy.integrate import quad

# 原始函數 f(x) = x^2 * sin(x)
def f(x):
    return x**2 * np.sin(x)

# 設定參數
m = 16  # 節點數
n = 4   # 三角多項式階數 S4
a, b = 0, 1  # 區間

# 建立等距節點
x_nodes = np.linspace(a, b, m, endpoint=False)
f_values = f(x_nodes)

# 定義三角基底函數 phi_k
def phi_k(k, x):
    if k == 0:
        return np.ones_like(x)
    elif k % 2 == 1:
        return np.cos(2 * np.pi * ((k + 1)//2) * x)
    else:
        return np.sin(2 * np.pi * (k//2) * x)

# 建立 A 矩陣
A = np.column_stack([phi_k(k, x_nodes) for k in range(2 * n + 1)])

# 最小平方解：解出係數 c
c, *_ = np.linalg.lstsq(A, f_values, rcond=None)

# 定義逼近函數 S4(x)
def S4(x):
    return sum(c[k] * phi_k(k, x) for k in range(2 * n + 1))

# 計算積分與誤差
integral_S4, _ = quad(S4, 0, 1)
true_integral, _ = quad(f, 0, 1)
difference = abs(integral_S4 - true_integral)
error = np.sum((f_values - A @ c) ** 2)

# 整齊輸出
print("=" * 56)
print(f"{'(b) ∫₀¹ S4(x) dx':40s} ≈ {integral_S4:> .10f}")
print(f"{'(c) ∫₀¹ x² sin(x) dx':40s} ≈ {true_integral:> .10f}")
print(f"{'(c) 積分誤差 |∫S4 - ∫f|':36s} ≈ {difference:> .10f}")
print(f"{'(d) 離散平方誤差 E(S4)':34s} ≈ {error:> .10f}")
print("=" * 56)
