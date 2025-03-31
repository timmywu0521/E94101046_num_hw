import numpy as np
from scipy.interpolate import lagrange

# 給定的數據點 (x, y = e^{-x})
x_data = np.array([0.3, 0.4, 0.5, 0.6])  # x 值
y_data = np.array([0.740818, 0.670320, 0.606531, 0.548812])  # y = e^{-x}

# 構造拉格朗日插值多項式（從 x 插值到 y）
poly = lagrange(x_data, y_data)

# 迭代反插值法求解 x = e^{-x}
def inverse_interpolation(poly, initial_guess=0.5, tol=1e-6, max_iter=100):
    x = initial_guess
    for i in range(max_iter):
        # 計算新近似值：x_new = P(e^{-x})
        y_current = np.exp(-x)
        x_new = poly(y_current)  # 求解 x_new = P(y_current)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return x_new, i+1

# 求解方程
solution, iterations = inverse_interpolation(poly)
print(f"迭代反插值法近似解: x = {solution:.8f}")
print(f"實際解驗證 (x - e^(-x)): {solution - np.exp(-solution):.8f}")
print(f"迭代次數: {iterations}")