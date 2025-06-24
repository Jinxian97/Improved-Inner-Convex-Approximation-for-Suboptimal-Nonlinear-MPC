import scipy.linalg as la
import numpy as np

def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 500
    eps = 1e-4

    for i in range(maxiter):
        Xn = A.T @ X @ A - A.T @ X @ B @ la.pinv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        if (abs(Xn - X)).max()  < eps:
            X = Xn
            break
        X = Xn

    return Xn

def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = la.pinv(B.T @ X @ B + R) @ (B.T @ X @ A)

    return X,K