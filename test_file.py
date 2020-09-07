depth = 20 #Previously 5 has worked well.
layers = 30 #5 works well.
size_classes = 2
lam = 2
time_step = 0.0001
simulate = False
verbose = False

from size_based_ecosystem import *

mass_vector = np.array([1, 20]) #np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats
obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)

res_start = logn


water_start = water_column(obj, res_start, layers = layers, replacement = lam, advection = 1, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj)
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start)
eco.population_setter(np.array([1, 0.1]) )
print(params.layered_attack[:, 1, :], eco.strategy_matrix)
print((eco.strategy_matrix*params.layered_attack[:, 1, :].T).T )
print((eco.strategy_matrix*params.layered_attack[:, 1, :].T) @ obj.M @ eco.strategy_matrix[1])

eco.one_actor_growth(0)
