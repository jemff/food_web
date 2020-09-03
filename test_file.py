depth = 20 #Previously 5 has worked well.
layers = 120 #5 works well.
log_size_range = 12 # 9.5 #8 did NOT work well.
size_classes = 2
t_end = 1
lam = 8
time_step = 0.0001
res_max = 15
simulate = False
verbose = False

from size_based_ecosystem import *

mass_vector = np.array([1, 20]) #np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats
obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)
print(logn, "Logn")

res_start = logn


water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, time_step = time_step, replacement = lam, advection = 1, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj)
eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, loss = 'constr')
eco.population_setter(np.array([1, 0.1]) )#, 1, 1, 1, 0.1]))
#eco.strategy_setter(np.sqrt(eco.strategy_matrix.flatten())) THis is for the L2 version... Quantum fish ahoy

print(eco.casadi_total_growth())