import math
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate

from PSO import PSO



def rosenbrock_function(x):
    """Rosenbrock's function"""
    f = 0
    dim = len(x)
    for i in range(dim-1):
        f += 100*(x[i]**2 - x[i+1])**2 + (x[i]-1)**2
    return f


# Initial Coefficients
pso_20 = PSO(func=rosenbrock_function, n_dim=20, pop=60, max_iter=30, lb=[-30]*20, ub=[30]*20, w=0.72984, c1=2.05, c2=2.05)
pso_20.run()

# Optimal Coeffiecients
pso_opt_20 = PSO(func=rosenbrock_function, n_dim=20, pop=60, max_iter=30, lb=[-30]*20, ub=[30]*20, w=-0.4736, c1=-0.9700, c2=3.7904)
pso_opt_20.run()

# Coefficients Optimized During Iterations
pso_auto_20 = PSO(func=rosenbrock_function, n_dim=20, pop=60, max_iter=30, lb=[-30]*20, ub=[30]*20, auto_coef=True)
pso_auto_20.run()
print('\nFOR 20 ITERATIONS\n')
# print('best_x (the global best value of every particle) is ', *pso_auto_20.gbest_x)

# Data
headers = ["Name", "Global Best", "Mean", "Standard Deviation"]
print(tabulate([['Initial', *pso_20.gbest_y, np.mean(pso_20.gbest_x), np.std(pso_20.gbest_x)],
                ['Optimal', *pso_opt_20.gbest_y, np.mean(pso_opt_20.gbest_x), np.std(pso_opt_20.gbest_x)],
                ['Auto', *pso_auto_20.gbest_y, np.mean(pso_auto_20.gbest_x), np.std(pso_auto_20.gbest_x)]], headers=headers))


# Plotting the global minimum value
plt.plot(pso_20.gbest_y_hist, '-r', label='Initial')
plt.plot(pso_opt_20.gbest_y_hist, '-g', label="Optimal")
plt.plot(pso_auto_20.gbest_y_hist, '-b', label="Auto")

plt.legend(loc="upper right")
plt.show()

