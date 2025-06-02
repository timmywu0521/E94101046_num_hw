import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import root_scalar
from scipy.linalg import solve
from tabulate import tabulate

h = 0.1
x_vals = np.linspace(0, 1, int(1 / h) + 1)

n = len(x_vals)

# (a) Shooting Method
def ode_system(x, Y):
    y, z = Y
    dy_dx = z
    dz_dx = -(x + 1)*z + 2*y + (1 - x**2)*np.exp(-x)
    return [dy_dx, dz_dx]

def shoot(s):
    sol = solve_ivp(ode_system, [0, 1], [1, s], t_eval=[1])
    return sol.y[0, -1] - 2

shoot_sol = root_scalar(shoot, bracket=[0, 10], method='brentq')
s_correct = shoot_sol.root
sol_full = solve_ivp(ode_system, [0, 1], [1, s_correct], t_eval=x_vals)
shooting_y = sol_full.y[0]

# (b) Finite Difference Method
A = np.zeros((n, n))
b = np.zeros(n)
A[0, 0] = 1
b[0] = 1
A[-1, -1] = 1
b[-1] = 2

for i in range(1, n - 1):
    x = x_vals[i]
    A[i, i - 1] = 1/h**2 - (x + 1)/(2*h)
    A[i, i] = -2/h**2 + 2
    A[i, i + 1] = 1/h**2 + (x + 1)/(2*h)
    b[i] = (1 - x**2)*np.exp(-x)

fdm_y = solve(A, b)

# (c) Variation Approach (Galerkin Method with 1 basis function)

# Define y0(x): satisfies boundary conditions y(0)=1, y(1)=2
def y0(x):
    return 1 + x  # linear interpolant between 1 and 2

# Define basis function φ1(x) that vanishes at boundaries
phi = lambda x: x * (1 - x)
dphi = lambda x: 1 - 2*x
d2phi = lambda x: -2 * np.ones_like(x)

# Residual operator L[φ]
def L_phi(x):
    return d2phi(x) + (x + 1)*dphi(x) - 2*phi(x)

# Right-hand side: f(x) - L[y0]
def rhs(x):
    return (1 - x**2)*np.exp(-x) - (0 + (x + 1)*1 - 2*y0(x))

# Compute integrals
A = quad(lambda x: L_phi(x)*phi(x), 0, 1)[0]
B = quad(lambda x: rhs(x)*phi(x), 0, 1)[0]

# Solve for coefficient c
c1 = B / A

# Construct solution at all x
variation_y = y0(x_vals) + c1 * phi(x_vals)

# 結果排版輸出
table_data = []
for i in range(n):
    row = [
        f"{x_vals[i]:.1f}",
        f"{shooting_y[i]:.8f}",
        f"{fdm_y[i]:.8f}",
        f"{variation_y[i]:.8f}"
    ]
    table_data.append(row)

headers = ["x", "Shooting Method", "Finite Difference", "Variation Approach"]
formatted_table = tabulate(table_data, headers=headers, tablefmt="grid")

print("=== BVP Solution Comparison ===")
print(formatted_table)
