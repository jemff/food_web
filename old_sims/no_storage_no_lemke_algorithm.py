depth = 30 #Previously 5 has worked well.
layers = 60 #5 works well.
segments = 1
size_classes = 2
lam = 2
simulate = False
verbose = True
l2 = False



from size_based_ecosystem import *
import pickle as pkl


mass_vector = np.array([20, 8000]) #np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats
obj = spectral_method(depth, layers, segments = segments)
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc = 6, scale = 6)
res_start = 3*norm_dist #0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers = layers*segments, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj, lam=0.2)
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers*segments, params, obj, water_start, l2 = l2, movement_cost=0)
eco.population_setter(np.array([2, 0.1]) )
OG_layered_attack = np.copy(eco.parameters.layered_attack)

day_interval = 192

time_step = (1/day_interval)*(1/365)
eco.heat_kernels[1] = eco.heat_kernels[0]
error = 1
strategies = []
population_list = []
resource_list = []
time = 0
prior_sol = quadratic_optimizer(eco)



strategy_list = []

periodic_layers = periodic_attack(params.layered_attack, day_interval=day_interval, minimum_attack=0.001, darkness_length = 2)
total_time_steps = 10 * day_interval  # Yup
for i in range(total_time_steps):
    current_foraging = foraging_gain_builder(eco)
    prior_sol = quadratic_optimizer(eco)
    x_res = (prior_sol[0:eco.populations.size * eco.layers]).reshape((eco.populations.size, -1))
    strategy_list.append(x_res)

    pop_old = np.copy(eco.populations)
    population_list.append(pop_old)
    eco.parameters.layered_attack = periodic_layers[i % day_interval]
    delta_pop = eco.total_growth(x_res)
    new_pop = delta_pop * time_step + eco.populations
    error = np.linalg.norm(new_pop - pop_old)

    eco.population_setter(eco.total_growth(x_res) * time_step + eco.populations)
    eco.strategy_setter(x_res)
    r_c = np.copy(eco.water.res_counts)
    resource_list.append(r_c)

#    eco.water.update_resources(consumed_resources=eco.consumed_resources(), time_step=time_step)
    print("I'm here")
    print(error, eco.populations, np.sum(eco.water.res_counts), time_step, new_pop - pop_old, time, i)
    time += time_step

with open('eco_big.pkl', 'wb') as f:
    pkl.dump(eco, f, pkl.HIGHEST_PROTOCOL)

with open('strategies_eco_big.pkl', 'wb') as f:
    pkl.dump(strategy_list, f, pkl.HIGHEST_PROTOCOL)

with open('population_eco_big.pkl', 'wb') as f:
    pkl.dump(population_list, f, pkl.HIGHEST_PROTOCOL)

with open('resource_eco_big.pkl', 'wb') as f:
    pkl.dump(resource_list, f, pkl.HIGHEST_PROTOCOL)

with open('periodic_layers_eco_big.pkl', 'wb') as f:
    pkl.dump(periodic_layers, f, pkl.HIGHEST_PROTOCOL)
