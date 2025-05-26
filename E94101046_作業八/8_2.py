import numpy as np

# 定義 f(x)
def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)

# 數值內積（使用梯形積分）
def inner_product(f1, f2, x):
    y = f1(x) * f2(x)
    return np.trapz(y, x)

# 積分區間 [-1, 1]
x = np.linspace(-1, 1, 1000)

# 基底函數
phi0 = lambda x: np.ones_like(x)
phi1 = lambda x: x
phi2 = lambda x: x**2

# 建立 Gram 矩陣 A
A = np.array([
    [inner_product(phi0, phi0, x), inner_product(phi0, phi1, x), inner_product(phi0, phi2, x)],
    [inner_product(phi1, phi0, x), inner_product(phi1, phi1, x), inner_product(phi1, phi2, x)],
    [inner_product(phi2, phi0, x), inner_product(phi2, phi1, x), inner_product(phi2, phi2, x)],
])

# 建立右邊向量 b
b = np.array([
    inner_product(f, phi0, x),
    inner_product(f, phi1, x),
    inner_product(f, phi2, x)
])

# 解聯立方程式 A * a = b
a = np.linalg.solve(A, b)

# 整齊輸出結果
print("=" * 50)
print("最小平方二次多項式係數為：")
print(f"{'a₀':<10s} = {a[0]:> .6f}")
print(f"{'a₁':<10s} = {a[1]:> .6f}")
print(f"{'a₂':<10s} = {a[2]:> .6f}")
print("-" * 50)
print(f"近似多項式為：P₂(x) = {a[0]:.6f} + {a[1]:.6f}·x + {a[2]:.6f}·x²")
print("=" * 50)
