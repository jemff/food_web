depth = 30 #Previously 5 has worked well.
layers = 120 #5 works well.
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
res_start = norm_dist #0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers = layers*segments, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj, lam=0.2)
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers*segments, params, obj, water_start, l2 = l2, movement_cost=0)
eco.population_setter(np.array([2, 0.1]) )
OG_layered_attack = np.copy(eco.parameters.layered_attack)

time_step = 1/192*1/365 #Time-step is half an hour.
eco.heat_kernels[1] = eco.heat_kernels[0]
error = 1
strategies = []
population_list = []
resource_list = []
time = 0
prior_sol = quadratic_optimizer(eco)


day_interval = 192
time_step = 1 / 365 * 1 / day_interval

resource_list = []
population_list = []
strategy_list = []

total_time_steps = 120 * day_interval  # Yup
time = 0
for i in range(total_time_steps):
    x_res = lemke_optimizer(eco)
    strategies.append(x_res)
    pop_old = np.copy(eco.populations)
    delta_pop = eco.total_growth(x_res)
    new_pop = delta_pop * time_step + eco.populations
    population_list.append(pop_old)
    error = np.linalg.norm(new_pop - pop_old)

    eco.population_setter(eco.total_growth(x_res) * time_step + eco.populations)
    eco.strategy_setter(x_res)
    r_c = np.copy(eco.water.res_counts)
    resource_list.append(r_c)

    print("I'm here")
    print(error, eco.populations, np.sum(eco.water.res_counts), time_step, new_pop - pop_old, time)
    time += time_step
    eco.parameters.layered_attack = 1/2*(1+np.cos(time*365*2*np.pi))*OG_layered_attack

with open('lemke_no_storage.pkl', 'wb') as f:
    pkl.dump(eco, f, pkl.HIGHEST_PROTOCOL)

with open('strategies_lemke_no_storage.pkl', 'wb') as f:
    pkl.dump(strategies, f, pkl.HIGHEST_PROTOCOL)

with open('population_lemke_no_storage.pkl', 'wb') as f:
    pkl.dump(population_list, f, pkl.HIGHEST_PROTOCOL)

with open('resource_lemke_no_storage.pkl', 'wb') as f:
    pkl.dump(resource_list, f, pkl.HIGHEST_PROTOCOL)