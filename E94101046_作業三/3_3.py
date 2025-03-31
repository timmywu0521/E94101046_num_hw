import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import fsolve, minimize_scalar

# 已知數據點
T = np.array([0, 3, 5, 8, 13])  # 時間 (秒)
D = np.array([0, 200, 375, 620, 990])  # 位置 (英尺)
V = np.array([75, 77, 80, 74, 72])  # 速度 (英尺/秒)

# 1. 使用 Cubic Hermite Spline 進行插值
hermite_spline = CubicHermiteSpline(T, D, V)

# 2. 預測 t = 10 秒時的 位置 D(10) 和 速度 V(10)
t_10 = 10
D_10 = hermite_spline(t_10)  # 位置
V_10 = hermite_spline.derivative()(t_10)  # 速度

print(f"a. D(10) = {D_10:.2f} 英尺, V(10) = {V_10:.2f} 英尺/秒")

# 3. 判斷何時超過 80.67 ft/s
speed_limit = 80.67

def speed_eq(t):
    return hermite_spline.derivative()(t) - speed_limit

try:
    t_exceed = fsolve(speed_eq, x0=5)  # 初始猜測為 5 秒
    t_exceed = t_exceed[(t_exceed >= 0) & (t_exceed <= 13)]  # 只保留有效範圍內的解
    if len(t_exceed) > 0:
        print(f"b. 第一次超過 55 mi/h (80.67 ft/s) 的時間: t = {t_exceed[0]:.2f} 秒")
    else:
        print("車輛從未超過 55 mi/h")
except:
    print("無法計算超速時間")

# 4. 預測最大速度
res = minimize_scalar(lambda t: -hermite_spline.derivative()(t), bounds=(0, 13), method='bounded')
t_max_speed = res.x
V_max = -res.fun  # 取反求最大值

print(f"c. 最大速度 = {V_max:.2f} 英尺/秒, 發生時間: t = {t_max_speed:.2f} 秒")
