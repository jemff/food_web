depth = 20  # Previously 5 has worked well.
layers = 100  # 5 works well.
lam = 2
simulate = False
verbose = True
l2 = False

from size_based_ecosystem import *
import pickle as pkl

mass_vector = np.array([20, 8000])  # np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats

obj = spectral_method(depth, layers - 1)  # This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc=2)
res_start = 3 * norm_dist  # 0.1*(1-obj.x/depth)
res_max = 10 * norm_dist

water_start = water_column(obj, res_start, layers=layers, resource_max=res_max, replacement=lam, advection=0,
                           diffusion=0)

params = ecosystem_parameters(mass_vector, obj)
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, l2=l2, movement_cost=0, time_step=10**(-6))
eco.population_setter(np.array([1, 0.001]) )

day_interval = 192
time_step = 1 / 365 * 1 / day_interval

resource_list = []
population_list = []
strategy_list = []

periodic_layers = periodic_attack(params.layered_attack, day_interval=day_interval, minimum_attack=0.001, darkness_length = 2)
print("Checkpoint 0")
reward_t, loss_t = reward_loss_time_dependent(eco, periodic_layers=periodic_layers)
print("Checkpoint 1")
total_time_steps = 120 * day_interval  # Yup
time = 0
for i in range(total_time_steps):
    current_reward = reward_t[i % day_interval]
    current_loss = loss_t[i % day_interval]
    current_foraging = foraging_gain_builder(eco)
    payoff_matrix = total_payoff_matrix_builder_memory_improved(eco, eco.populations,
                                                                total_reward_matrix=current_reward,
                                                                total_loss_matrix=current_loss,
                                                                foraging_gain=current_foraging)

    prior_sol = quadratic_optimizer(eco, payoff_matrix=payoff_matrix)
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

    eco.water.update_resources(consumed_resources=eco.consumed_resources(), time_step=time_step)
    print("I'm here")
    print(error, eco.populations, np.sum(eco.water.res_counts), time_step, new_pop - pop_old, time, i)
    time += time_step

with open('eco_long_night_HD_narrow_kernel.pkl', 'wb') as f:
    pkl.dump(eco, f, pkl.HIGHEST_PROTOCOL)

with open('strategies_eco_long_night_HD_narrow_kernel.pkl', 'wb') as f:
    pkl.dump(strategy_list, f, pkl.HIGHEST_PROTOCOL)

with open('population_eco_long_night_HD_narrow_kernel.pkl', 'wb') as f:
    pkl.dump(population_list, f, pkl.HIGHEST_PROTOCOL)

with open('resource_eco_long_night_HD_narrow_kernel.pkl', 'wb') as f:
    pkl.dump(resource_list, f, pkl.HIGHEST_PROTOCOL)

with open('rewards_eco_long_night_HD_narrow_kernel.pkl', 'wb') as f:
    pkl.dump(reward_t, f, pkl.HIGHEST_PROTOCOL)

with open('losses_eco_long_night_HD_narrow_kernel.pkl', 'wb') as f:
    pkl.dump(loss_t, f, pkl.HIGHEST_PROTOCOL)

with open('periodic_layers_eco_long_night_HD_narrow_kernel.pkl', 'wb') as f:
    pkl.dump(periodic_layers, f, pkl.HIGHEST_PROTOCOL)
