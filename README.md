# Towards Improved Performance of Inner Convex Approximation for Suboptimal Nonlinear MPC

## Abstract of This Paper

Inner convex approximation is a compelling method that enables the real-time implementation of suboptimal nonlinear model predictive control (MPC). However, it suffers from a slow convergence rate, which prevents suboptimal MPC from achieving better performance within a specific sample time. To address this issue, we first reformulate the conventional inner convex approximation procedure as a root-finding problem for a nonlinear equation. Then, under mild assumptions, a comprehensive functional analysis is performed on the derived nonlinear equation, focusing on its continuity, differentiability, and the invertibility of the Jacobian matrix. Building on this analysis, we propose an improved algorithm that applies Broyden's method to accelerate the root-finding procedure of this derived nonlinear equation, thereby enhancing the convergence rate of the conventional inner convex approximation method. We also provide a detailed analysis of the proposed algorithm's convergence properties and computational complexity, showing that it achieves a locally superlinear convergence rate without devoting much additional computational effort. Simulation experiments are performed in an obstacle avoidance scenario, and the results are compared to the conventional inner convex approximation method to assess the effectiveness and advantages of the proposed approach.

## Files

This repository contains a collection of Python scripts related to data preprocessing and algorithm implementations. Below is an overview of the files included:

- `algorithm1.py`: Main Suboptimal nonlinear MPC Scheme, which involves Algorithm 2 as its sub-procedure.
- `algorithm1_full_NLP.py` : Optimal nonlinear MPC-based algorithm implementation, which involves IPOPT for solving the optimal solution of each OCP.
- `algorithm1_onestep.py` : Suboptimal nonlinear MPC Scheme with Algorithm 2 executed  only once.
- `algorithm2.py` : Algorithm 2 (main contribution).
- `dlqr.py` : Discrete-time linear quadratic regular algorithm.
- `Dparameterin.py` : Parameter input script.

## Usage

1. Clone the repository to your local machine.
2. Install required Python dependencies.
3. Run the scripts as needed for your specific use case.

## Notes

- All files are written in Python.
- Requirement: Casadi.
