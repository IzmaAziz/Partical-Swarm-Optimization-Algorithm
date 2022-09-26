import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from PSO import PSO


def griewank_function(x):
    """Griewank's function multimodal, symmetric, inseparable """
    partA = 0
    partB = 1
    dim = len(x)

    for i in range(dim):
        partA += x[i]**2
        partB *= math.cos(float(x[i]) / math.sqrt(i+1))

    return 1 + (float(partA)/4000.0) - float(partB)  

# Initial Coefficients
pso_20 = PSO(func=griewank_function, n_dim=20, pop=90, max_iter=30, lb=[-30]*20, ub=[30]*20, w=0.72984, c1=2.05, c2=2.05)
pso_20.run()

# Optimal Coeffiecients
pso_opt_20 = PSO(func=griewank_function, n_dim=20, pop=90, max_iter=30, lb=[-30]*20, ub=[30]*20, w=0.8, c1=0.5, c2=0.5)
pso_opt_20.run()

# Coefficients Optimized During Iterations
pso_auto_20 = PSO(func=griewank_function, n_dim=20, pop=90, max_iter=30, lb=[-30]*20, ub=[30]*20, auto_coef=True)
pso_auto_20.run()
print('\nFOR 20 ITERATIONS\n')
# print('best_x (the global best value of every particle) is ', *pso_auto_20.gbest_x)

# Data
headers = ["Name", "Global Best", "Mean", "Standard Deviation"]
print(tabulate([['Initial', *pso_20.gbest_y, np.mean(pso_20.gbest_x), np.std(pso_20.gbest_x)],
                ['Optimal', *pso_opt_20.gbest_y, np.mean(pso_opt_20.gbest_x), np.std(pso_opt_20.gbest_x)],
                ['Auto', *pso_auto_20.gbest_y, np.mean(pso_auto_20.gbest_x), np.std(pso_auto_20.gbest_x)]], headers=headers))


# Plotting the global minimum value
plt.subplot(1,2,1)
plt.plot(pso_20.gbest_y_hist, '-r', label='Initial')
plt.plot(pso_opt_20.gbest_y_hist, '-g', label="Optimal")
plt.plot(pso_auto_20.gbest_y_hist, '-b', label="Auto")
plt.title("FOR D=20")
plt.legend(loc="upper right")




# Initial Coefficients
pso_50 = PSO(func=griewank_function, n_dim=50, pop=150, max_iter=30, lb=[-30]*50, ub=[30]*50, w=0.72984, c1=2.05, c2=2.05)
pso_50.run()

# Optimal Coeffiecients
pso_opt_50 = PSO(func=griewank_function, n_dim=50, pop=150, max_iter=30, lb=[-30]*50, ub=[30]*50, w=0.8, c1=0.5, c2=0.5)
pso_opt_50.run()

# Coefficients Optimized During Iterations
pso_auto_50 = PSO(func=griewank_function, n_dim=50, pop=150, max_iter=30, lb=[-30]*50, ub=[30]*50, auto_coef=True)
pso_auto_50.run()
print('\nFOR 50 ITERATIONS\n')


# Data
headers = ["Name", "Global Best", "Mean", "Standard Deviation"]
print(tabulate([['Initial', *pso_50.gbest_y, np.mean(pso_50.gbest_x), np.std(pso_50.gbest_x)],
                ['Optimal', *pso_opt_50.gbest_y, np.mean(pso_opt_50.gbest_x), np.std(pso_opt_50.gbest_x)],
                ['Auto', *pso_auto_50.gbest_y, np.mean(pso_auto_50.gbest_x), np.std(pso_auto_50.gbest_x)]], headers=headers))


# Plotting the global minimum value
plt.subplot(1,2,2)
plt.plot(pso_50.gbest_y_hist, '-r', label='Initial')
plt.plot(pso_opt_50.gbest_y_hist, '-g', label="Optimal")
plt.plot(pso_auto_50.gbest_y_hist, '-b', label="Auto")
plt.title("FOR D=50")

plt.legend(loc="upper right")
plt.show()
