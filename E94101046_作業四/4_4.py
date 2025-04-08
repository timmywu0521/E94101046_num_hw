import numpy as np

# Composite Simpson's Rule function
def composite_simpson(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (h / 3) * (y[0] + 2 * sum(y[2:n:2]) + 4 * sum(y[1:n:2]) + y[n])

# Part (a): ∫₀¹ x^(-1/4) * sin(x) dx
def f_a(x):
    return np.where(x == 0, 0, x**(-1/4) * np.sin(x))  # avoid divide by zero

# Part (b): ∫₁^∞ x^(-4) * sin(x) dx
# After substitution: -∫₀¹ t² * sin(1/t) dt
def f_b(t):
    return np.where(t == 0, 0, t**2 * np.sin(1 / t))  # avoid divide by zero

# Parameters
n = 4
eps = 1e-6  # small epsilon to avoid singularity at 0

# Calculate
integral_a = composite_simpson(f_a, eps, 1, n)
integral_b = composite_simpson(f_b, eps, 1, n)

# Print results
print("Approximation of ∫₀¹ x^(-1/4) * sin(x) dx =", integral_a)
print("Approximation of ∫₁^∞ x^(-4) * sin(x) dx  =", integral_b)
