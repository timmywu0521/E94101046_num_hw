import numpy as np
import math  # 替換 np.math

# 給定數據點
x_vals = np.array([0.698, 0.733, 0.768, 0.803])
y_vals = np.array([0.7661, 0.7432, 0.7193, 0.6946])

# 目標插值點
x_target = 0.750

# Lagrange 插值函數
def lagrange_interpolation(x_vals, y_vals, x_target):
    n = len(x_vals)
    P_x = 0  # 插值多項式值
    
    for i in range(n):
        # 計算 L_i(x)
        L_i = 1
        for j in range(n):
            if i != j:
                L_i *= (x_target - x_vals[j]) / (x_vals[i] - x_vals[j])
        
        # 加總貢獻值
        P_x += y_vals[i] * L_i

    return P_x

# 計算誤差邊界
def error_bound(n, x_target, x_vals, max_derivative_value):
    # 計算誤差邊界公式
    factor = 1
    for i in range(n):
        factor *= np.abs(x_target - x_vals[i])
    return max_derivative_value * factor / math.factorial(n)

# 計算 cos(x) 的最大導數 (第二導數的最大值)
# cos(x) 的第二導數為 -cos(x)，因此最大值為 |cos(x)| 在插值區間的最大值
max_derivative_value = np.max(np.abs(np.cos(x_vals)))

# 計算不同次數的插值和誤差邊界
results = {}
for degree in range(1, 4):  # 使用 1 到 4 次的插值
    # 插值多項式
    interpolation_result = lagrange_interpolation(x_vals[:degree+1], y_vals[:degree+1], x_target)
    # 計算誤差邊界
    error = error_bound(degree, x_target, x_vals[:degree+1], max_derivative_value)
    
    # 儲存結果
    results[f"Degree {degree}"] = {
        "Interpolation Result": interpolation_result,
        "Error Bound": error
    }

# 印出結果
for key, value in results.items():
    print(f"{key}:")
    print(f"  Interpolation Result: {value['Interpolation Result']:.6f}")
    print(f"  Error Bound: {value['Error Bound']:.6e}")
