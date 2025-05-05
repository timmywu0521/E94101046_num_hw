import numpy as np
from scipy.linalg import lu_solve, lu_factor

# 定義係數矩陣 A 和常數項向量 b
A = np.array([[ 3, -1,  0,  0],
              [-1,  3, -1,  0],
              [ 0, -1,  3, -1],
              [ 0,  0, -1,  3]], dtype=float)

b = np.array([2, 3, 4, 1], dtype=float)

# 使用 Crout 方法進行 LU 分解
lu, piv = lu_factor(A)

# 解方程組 Ax = b
x = lu_solve((lu, piv), b)

# 印出結果
print("解向量 x 為：")
for i, val in enumerate(x, 1):
    print(f"x{i} = {val:.4f}")
