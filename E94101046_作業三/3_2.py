import numpy as np

# 给定的數據點
x_data = np.array([0.3, 0.4, 0.5, 0.6])  # x 值
y_data = np.array([0.740818, 0.670320, 0.606531, 0.548812])  # y = e^{-x} 的值

# 二次拉格朗日反插值（使用最近的 3 個点）
def inverse_interpolation(y, x_nodes, y_nodes):
    n = len(x_nodes)
    # 找到 y 所在的區間
    for i in range(n - 2):
        if y_nodes[i+1] <= y <= y_nodes[i]:
            # 使用 x[i], x[i+1], x[i+2] 構造二次插值多项式
            x0, x1, x2 = x_nodes[i], x_nodes[i+1], x_nodes[i+2]
            y0, y1, y2 = y_nodes[i], y_nodes[i+1], y_nodes[i+2]
            # 計算拉格朗日函数
            L0 = ((y - y1) * (y - y2)) / ((y0 - y1) * (y0 - y2))
            L1 = ((y - y0) * (y - y2)) / ((y1 - y0) * (y1 - y2))
            L2 = ((y - y0) * (y - y1)) / ((y2 - y0) * (y2 - y1))
            return x0 * L0 + x1 * L1 + x2 * L2
    # 如果 y 超出範圍，返回 None（後續用二分法處理）
    return None

# 二分法求解 x = e^{-x}
def bisection_method(a, b, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        c = (a + b) / 2
        if np.exp(-c) - c == 0 or (b - a) / 2 < tol:
            return c
        if (np.exp(-a) - a) * (np.exp(-c) - c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2


initial_guess = 0.5  # 初始猜测
y_guess = initial_guess
x_interp = inverse_interpolation(y_guess, x_data, y_data)

if x_interp is not None:

    if 0.3 <= x_interp <= 0.6:
        solution = x_interp
    else:

        solution = bisection_method(0.3, 0.6)
else:

    solution = bisection_method(0.3, 0.6)

print(f"Approximate solution to x = e^(-x): {solution:.6f}")