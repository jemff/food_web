depth = 20 #Previously 5 has worked well.
layers = 50 #5 works well.
size_classes = 2
lam = 2
time_step = 0.0001
simulate = False
verbose = False
l2 = False
from size_based_ecosystem import *

mass_vector = np.array([1, 400]) #np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats
obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc = 2)
res_start = 3*norm_dist #0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj)
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, l2=l2, verbose=True, movement_cost=0)
eco.population_setter(np.array([1, 0.0000001]) )
print(eco.heat_kernels.shape)
old_sol = quadratic_optimizer(eco, eco.strategy_matrix)
strategy_matrix = old_sol[0:eco.populations.size*eco.layers].reshape((eco.populations.size, -1))
eco.strategy_setter(strategy_matrix)
print(strategy_matrix[0], eco.populations.size)
print(eco.one_actor_growth(0, solve_mode=False))
opt_strat = eco.one_actor_growth(0).flatten()
print(opt_strat)
strategy_matrix[0] = opt_strat
print(eco.one_actor_growth(0, solve_mode=False))