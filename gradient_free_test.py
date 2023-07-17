import numpy as np
from gradient_free_optimizers import SimulatedAnnealingOptimizer
def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x *x * x + y * y + np.sin(y))
    
import time
start_time = time.time()
for i  in range(100):



    search_space = {
        "x": np.arange(-10, 10, 0.1),
        "y": np.arange(-10, 10, 0.1),
    }

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=1.2,
        annealing_rate=0.99,
    )
    opt.search(sphere_function, n_iter=100,verbosity=[])
print(opt.best_para)
end_time = time.time()
print('elapsed time:', end_time - start_time)
    # print(opt.best_para['x'],',',opt.best_para['y'])