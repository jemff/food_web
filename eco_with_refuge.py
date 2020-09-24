depth = 20 #Previously 5 has worked well.
layers = 50 #5 works well.
size_classes = 2
lam = 2
simulate = False
verbose = True
l2 = False



from size_based_ecosystem import *
import pickle as pkl


mass_vector = np.array([1, 20, 400, 8000]) #np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats
obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc = 2)
res_start = 3*norm_dist #0.1*(1-obj.x/depth)
res_max = 15*norm_dist

water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj)
params.handling_times = np.zeros(4)

eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, l2 = l2, movement_cost=0)
eco.population_setter(np.array([1, 0.0000001, 0.0000001, 0.0000001]) )

day_interval = 24*8
time_step = 1/365*1/day_interval

periodic_layers = periodic_attack(params.layered_attack, day_interval = day_interval)
print("Checkpoint 0")
reward_t, loss_t = reward_loss_time_dependent(eco, periodic_layers=periodic_layers)
print("Checkpoint 1")