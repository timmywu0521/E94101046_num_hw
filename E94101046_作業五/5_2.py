import numpy as np

# 微分方程組的右邊函數
def f(t, u):
    u1, u2 = u
    du1 = 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)
    du2 = -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)
    return np.array([du1, du2])

# 解析解
def exact_solution(t):
    u1 = 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)
    u2 = -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)
    return np.array([u1, u2])

# Runge-Kutta 4 方法
def runge_kutta_4(f, u0, t0, t_end, h):
    t_values = [t0]
    u_values = [u0]
    
    t = t0
    u = u0.copy()
    
    while t < t_end:
        if t + h > t_end:
            h = t_end - t  # 避免超出範圍
        k1 = f(t, u)
        k2 = f(t + h/2, u + h/2 * k1)
        k3 = f(t + h/2, u + h/2 * k2)
        k4 = f(t + h, u + h * k3)
        
        u += (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t += h
        t_values.append(t)
        u_values.append(u.copy())
        
    return np.array(t_values), np.array(u_values)

# 初始條件與時間範圍
u0 = np.array([4/3, 2/3])
t0 = 0
t_end = 1

# 執行並比較不同 h
for h in [0.1, 0.05]:
    t_values, u_values = runge_kutta_4(f, u0, t0, t_end, h)
    exact_values = np.array([exact_solution(t) for t in t_values])
    error = np.abs(u_values - exact_values)

    print(f"\n===== RK4 結果 (h = {h}) =====")
    print(f"{'t':>8} | {'u1_RK4':>18} | {'u1_exact':>18} | {'err_u1':>12} || {'u2_RK4':>18} | {'u2_exact':>18} | {'err_u2':>12}")
    print("-"*126)
    for t, num, ex, err in zip(t_values, u_values, exact_values, error):
        print(f"{t:8.2f} | {num[0]:18.6f} | {ex[0]:18.6f} | {err[0]:12.2e} || {num[1]:18.6f} | {ex[1]:18.6f} | {err[1]:12.2e}")
