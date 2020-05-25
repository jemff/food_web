depth = 10 #Previously 5 has worked well.
layers = 40 #5 works well.
log_size_range = 12 # 9.5 #8 did NOT work well.
size_classes = 3
t_end = 1
time_step = 0.0001
lam = 3
res_max = 15
simulate = False
verbose = False

from size_based_ecosystem import *

mass_vector = np.array([1, 400, 1000])
from scipy import stats
obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)
print(logn, "Logn")

res_start = logn

water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, time_step = time_step, replacement = lam, advection = 0.1, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj)
eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, loss = 'constr')
eco.population_setter(np.array([1, 0.5, 0]))

print(eco.one_actor_growth(eco.strategy_matrix.flatten(), 1))
print(eco.one_actor_growth(eco.strategy_replacer(eco.strategy_matrix[0], 1, eco.strategy_matrix.flatten()), 1), "What if we moved to the prey??")
seq_nash = sequential_nash(eco)
