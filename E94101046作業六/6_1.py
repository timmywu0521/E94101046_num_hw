import numpy as np

# 定義係數矩陣 A 和常數向量 b
A = np.array([
    [1.19, 2.11, -100, 1],
    [14.2, -0.112, 12.2, -1],
    [0, 100, -99.9, 1],
    [15.3, 0.110, -13.1, -1]
], dtype=float)

b = np.array([1.12, 3.44, 2.15, 4.16], dtype=float)

# 執行 Gaussian Elimination with Partial Pivoting
n = len(b)
for i in range(n):
    # Pivoting: 找出目前 column 最大值的 row
    max_row = np.argmax(abs(A[i:n, i])) + i
    if i != max_row:
        A[[i, max_row]] = A[[max_row, i]]
        b[i], b[max_row] = b[max_row], b[i]

    # Elimination: 把下面的 row 變成 0
    for j in range(i+1, n):
        factor = A[j, i] / A[i, i]
        A[j, i:] -= factor * A[i, i:]
        b[j] -= factor * b[i]

# 回代求解
x = np.zeros(n)
for i in range(n-1, -1, -1):
    x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

# 印出解
for idx, val in enumerate(x, start=1):
    print(f"x{idx} = {val:.6f}")
