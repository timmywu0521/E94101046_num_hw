import numpy as np

# --- 定義係數矩陣 A 和常數向量 b ---
A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 1, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# --- Jacobi Method ---
def jacobi(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for i in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new
        x = x_new
    return x

# --- Gauss-Seidel Method ---
def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new
        x = x_new
    return x

# --- SOR Method ---
def sor(A, b, w=1.25, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - w) * x[i] + w * (b[i] - sigma) / A[i, i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new
        x = x_new
    return x

# --- Conjugate Gradient Method (hand-coded) ---
def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x

# --- 執行各方法並印出解 ---
x_jacobi = jacobi(A, b)
x_gs = gauss_seidel(A, b)
x_sor = sor(A, b, w=1.25)
x_cg = conjugate_gradient(A, b)

print("Jacobi solution:         ", x_jacobi)
print("Gauss-Seidel solution:   ", x_gs)
print("SOR solution:            ", x_sor)
print("Conjugate Gradient (CG): ", x_cg)
