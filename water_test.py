depth = 10 #Previously 5 has worked well.
layers = 40 #5 works well.
log_size_range = 12 # 9.5 #8 did NOT work well.
size_classes = 2
t_end = 1
time_step = 0.00001
lam = 1
res_max = 1
simulate = True
verbose = False

from size_based_ecosystem import *

obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!


A, one, bounds = constraint_builder(obj.M, size_classes)
from scipy import stats as stats
logn = stats.lognorm.pdf(obj.x, 1, 0)
print(logn, "Logn")

res_start = logn

water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, time_step = time_step, replacement = lam, advection = 0.1, diffusion = 0.0)


if simulate is True:
    time_span = np.linspace(0, t_end, int(t_end/time_step))
    for t in time_span:
        water_start.update_resources()
        print(water_start.res_counts)