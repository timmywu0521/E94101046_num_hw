import numpy as np

# 定義函數
def f(x):
    return np.exp(x) * np.sin(4 * x)

# 設定積分範圍與步長
a = 1
b = 2
h = 0.1
n = int((b - a) / h)

# 建立節點
x = np.linspace(a, b, n + 1)

# Composite Trapezoidal Rule
def composite_trapezoidal(f, x, h):
    return (h / 2) * (f(x[0]) + 2 * np.sum(f(x[1:-1])) + f(x[-1]))

# Composite Simpson's Rule
def composite_simpson(f, x, h):
    if len(x) % 2 == 1:  # n 必須為偶數 => 節點數 x 要為奇數
        return (h / 3) * (f(x[0]) +
                          4 * np.sum(f(x[1:-1:2])) +
                          2 * np.sum(f(x[2:-2:2])) +
                          f(x[-1]))
    else:
        raise ValueError("Simpson's Rule requires an even number of intervals.")

# Composite Midpoint Rule
def composite_midpoint(f, a, b, h):
    n = int((b - a) / h)
    midpoints = a + h * (np.arange(n) + 0.5)
    return h * np.sum(f(midpoints))

# 計算各種方法的結果
trapz_result = composite_trapezoidal(f, x, h)
simpson_result = composite_simpson(f, x, h)
midpoint_result = composite_midpoint(f, a, b, h)

# 輸出結果
print("Composite Trapezoidal Rule: {:.6f}".format(trapz_result))
print("Composite Simpson's Rule:   {:.6f}".format(simpson_result))
print("Composite Midpoint Rule:    {:.6f}".format(midpoint_result))
