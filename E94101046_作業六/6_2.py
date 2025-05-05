import numpy as np

# 定義矩陣 A
A = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
])

# 計算反矩陣
A_inv = np.linalg.inv(A)

# 印出結果
np.set_printoptions(precision=4, suppress=True)  # 設定小數精度與抑制科學記號
print("矩陣 A 的反矩陣為：")
print(A_inv)
