from scipy.optimize import minimize

def objective_func(x):
    # Define your objective function here
    return (x[0] * x[0] + x[1] * x[1])

# Initial guess for optimization
initial_guess = [5.0,5.0]
import time
start_time = time.time()
# Run Nelder-Mead optimization
for i  in range(1000):
    result = minimize(objective_func, initial_guess, method='SLSQP')


# Nelder-Mead: The downhill simplex algorithm.
# Powell: A modified version of the Nelder-Mead method that uses conjugate directions to find a minimum.
# CG: Nonlinear conjugate gradient method.
# BFGS: Quasi-Newton method (Broyden-Fletcher-Goldfarb-Shanno).
# L-BFGS-B: Limited-memory BFGS with bound constraints.
# TNC: Truncated Newton algorithm with bounds and constraints support.
# COBYLA: Constrained optimization by linear approximation for constrained or unconstrained problems.
    

# Extract the optimized solution
optimized_solution = result.x
end_time = time.time()
print('elapsed time:', end_time - start_time)
print(optimized_solution)