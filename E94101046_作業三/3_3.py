import numpy as np
import scipy.interpolate as spi

# 給定的數據
T = np.array([0, 3, 5, 8, 13])  # 時間 (秒)
D = np.array([0, 200, 375, 620, 990])  # 距離 (英尺)
V = np.array([75, 77, 80, 74, 72])  # 速度 (英尺/秒)

# 使用 Cubic Hermite Spline 進行插值
hermite_spline = spi.CubicHermiteSpline(T, D, V)

# (a) 計算 t = 10 時的位置和速度
t_10 = 10
D_10 = hermite_spline(t_10)  # 位置
V_10 = hermite_spline.derivative()(t_10)  # 速度

# (b) 找出速度超過 80.6 ft/s 的最小時間
speed_limit = 80.6  # 55 mi/hr 轉換為 ft/s
t_vals = np.linspace(min(T), max(T), 5000)  # 細分時間區間
V_vals = hermite_spline.derivative()(t_vals)  # 計算速度

# 找到第一個超過 80.6 ft/s 的時間點
t_exceed = t_vals[np.where(V_vals > speed_limit)[0][0]] if np.any(V_vals > speed_limit) else None

# (c) 找到預測的最大速度
max_speed_index = np.argmax(V_vals)
max_speed = V_vals[max_speed_index]
max_speed_time = t_vals[max_speed_index]

# 顯示結果
print("(a) t = 10 時的位置: {:.2f} ft".format(D_10))
print("    t = 10 時的速度: {:.2f} ft/s".format(V_10))
if t_exceed:
    print("(b) 第一次超過 80.6 ft/s 的時間: {:.2f} s".format(t_exceed))
else:
    print("(b) 車輛未超過 80.6 ft/s")
print("(c) 預測的最大速度: {:.2f} ft/s 發生在 t = {:.2f} s".format(max_speed, max_speed_time))