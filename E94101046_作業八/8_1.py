
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

# Given data
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# === (a) Quadratic Fit: y = ax^2 + bx + c ===
X_poly = np.vstack([x**2, x, np.ones(len(x))]).T
coeffs_poly, _, _, _ = lstsq(X_poly, y, rcond=None)
y_poly_pred = X_poly @ coeffs_poly
error_poly = np.sum((y - y_poly_pred)**2)

a2, a1, a0 = coeffs_poly
print("=== (a) Quadratic Fit ===")
print(f"y = {a2:.4f} * x^2 + {a1:.4f} * x + {a0:.4f}")
print(f"Error = {error_poly:.5f}\n")

# === (b) Exponential Fit: y = b * e^(a*x) ===
X_exp = np.vstack([x, np.ones(len(x))]).T
log_y = np.log(y)
coeffs_exp, _, _, _ = lstsq(X_exp, log_y, rcond=None)
a_exp, log_b_exp = coeffs_exp
b_exp = np.exp(log_b_exp)
y_exp_pred = b_exp * np.exp(a_exp * x)
error_exp = np.sum((y - y_exp_pred)**2)

print("=== (b) Exponential Fit ===")
print(f"y = {b_exp:.4f} * e^({a_exp:.4f} * x)")
print(f"Error = {error_exp:.5f}\n")

# === (c) Power Fit: y = b * x^n ===
X_pow = np.vstack([np.log(x), np.ones(len(x))]).T
log_y_pow = np.log(y)
coeffs_pow, _, _, _ = lstsq(X_pow, log_y_pow, rcond=None)
n_pow, log_b_pow = coeffs_pow
b_pow = np.exp(log_b_pow)
y_pow_pred = b_pow * x**n_pow
error_pow = np.sum((y - y_pow_pred)**2)

print("=== (c) Power Fit ===")
print(f"y = {b_pow:.4f} * x^{n_pow:.4f}")
print(f"Error = {error_pow:.5f}")
