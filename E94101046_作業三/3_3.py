import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
from scipy.optimize import root_scalar, minimize_scalar

# 已知數據點
T = np.array([0, 3, 5, 8, 13])  # 時間 (秒)
D = np.array([0, 200, 375, 620, 990])  # 位置 (英尺)
V = np.array([75, 77, 80, 74, 72])  # 速度 (英尺/秒)

# 使用 PCHIP (分段三次 Hermite 插值) 構建插值函數
hermite_position = PchipInterpolator(T, D)  # 位置插值函數
hermite_velocity = PchipInterpolator(T, V)  # 速度插值函數

# 計算 t = 10 時的位置和速度
t_pred = 10
D_pred = hermite_position(t_pred)
V_pred = hermite_velocity(t_pred)

# 計算速度的導數來獲得加速度函數
hermite_acceleration = hermite_velocity.derivative()

# 計算 t = 10 時的加速度
A_pred = hermite_acceleration(t_pred)

# 計算從 t = 10 到 t = 11 的距離變化量 ΔD
delta_D, _ = quad(hermite_velocity, 10, 11)

# 速度限制 (55 mi/h 轉換為 ft/s)
speed_limit = 55 * 5280 / 3600  # 80.67 ft/s

# 檢查速度最大值是否超過 80.67 ft/s
max_velocity = max(hermite_velocity(T))

t_exceed = None  # 預設為未超速
if max_velocity > speed_limit:
    # 定義方程式 V(t) - speed_limit = 0
    def speed_exceeds_limit(t):
        return hermite_velocity(t) - speed_limit

    # 在 [0, 13] 秒內尋找最早超過速度限制的時間
    solution = root_scalar(speed_exceeds_limit, bracket=[0, 13], method='brentq')
    t_exceed = solution.root if solution.converged else None

# **計算預測最大速度**
def negative_velocity(t):
    return -hermite_velocity(t)

# 在 [0, 13] 內尋找最大速度點
max_speed_result = minimize_scalar(negative_velocity, bounds=(0, 13), method='bounded')

# 最大速度及對應時間
t_max_speed = max_speed_result.x
max_speed = -max_speed_result.fun

# **(a) 預測 t = 10 時的位置與速度**
print("(a) 預測 t = 10 秒時的位置與速度：")
print(f"位置 D({t_pred}) ≈ {D_pred:.2f} 英尺")
print(f"速度 V({t_pred}) ≈ {V_pred:.2f} 英尺/秒")
print()

# **(b) 判斷是否超過 55 mi/h 的速度限制**
print("(b) 車輛是否超過 55 mi/h 速度限制？")
if t_exceed is not None:
    print(f"車輛首次超過 55 mi/h (80.67 ft/s) 的時間為 t ≈ {t_exceed:.2f} 秒")
else:
    print("車輛在整個時間範圍內未曾超過 55 mi/h 的速度限制")
print()

# **(c) 預測車輛的最大速度**
print("(c) 預測車輛的最大速度：")
print(f"車輛的預測最大速度為 V_max ≈ {max_speed:.2f} 英尺/秒，發生於 t ≈ {t_max_speed:.2f} 秒")
