
import sys
sys.path.append('/home/jaem/projects/food_web/')
from size_based_ecosystem import *
import pickle as pkl
import siconos.numerics as sn
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def lemke_optimizer(eco, payoff_matrix = None):
    A = np.zeros((eco.populations.size, eco.populations.size * eco.layers))
    for k in range(eco.populations.size):
        A[k, k * eco.layers:(k + 1) * eco.layers] = -1

    q = np.zeros(eco.populations.size + eco.populations.size * eco.layers)
    q[eco.populations.size * eco.layers:] = -1
    q = q.reshape(-1, 1)
    if payoff_matrix is None:
        payoff_matrix = total_payoff_matrix_builder(eco)

    H = np.block([[-payoff_matrix, A.T], [-A, np.zeros((A.shape[0], eco.populations.size))]])
    lcp = sn.LCP(H, q)
    ztol = 1e-8

    #solvers = [sn.SICONOS_LCP_PGS, sn.SICONOS_LCP_QP,
    #           sn.SICONOS_LCP_LEMKE, sn.SICONOS_LCP_ENUM]

    z = np.zeros((eco.layers*2+eco.populations.size,), np.float64)
    w = np.zeros_like(z)
    options = sn.SolverOptions(sn.SICONOS_LCP_PIVOT)
    #sn.SICONOS_IPARAM_MAX_ITER = 10000000
    options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = 5000000

    info = sn.linearComplementarity_driver(lcp, z, w, options)
    print(sn.lcp_compute_error(lcp,z,w, ztol), "Error")
    print(info, "Info")
    return z

depth = 30  # Previously 5 has worked well.
layers = 60  # 5 works well.
segments = 1
size_classes = 2
lam = 2
simulate = False
verbose = True
l2 = False

mass_vector = np.array([20, 8000])  # np.array([1, 30, 300, 400, 800, 16000])

obj = spectral_method(depth, layers, segments=segments)
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc=1, scale=4)
res_start = 3 * norm_dist  # 0.1*(1-obj.x/depth)
res_max = 10 * norm_dist

water_start = water_column(obj, res_start, layers=layers * segments, resource_max=res_max, replacement=lam, advection=0,
                           diffusion=0)

params = ecosystem_parameters(mass_vector, obj, lam=0.4, min_attack_rate = 10**(-7))
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers * segments, params, obj, water_start, l2=l2, movement_cost=0)
eco.population_setter(np.array([2, 0.1]))
OG_layered_attack = np.copy(eco.parameters.layered_attack)

#eco.heat_kernels[1] = eco.heat_kernels[0]
eco.dirac_delta_creator()
#eco.heat_kernels[1] = eco.heat_kernels[0]


error = 1
strategies = []
population_list = []
resource_list = []
time = 0
prior_sol = lemke_optimizer(eco)
print(prior_sol)
#plt.plot(eco.spectral.x, prior_sol[60:120])
plt.show()

day_interval = 96
time_step = (1 / 365) * (1 / day_interval)

strategy_list = []

periodic_layers = periodic_attack(params.layered_attack, day_interval=day_interval, minimum_attack=0.0001,
                                  darkness_length=2)
reward_t, loss_t = reward_loss_time_dependent(eco, periodic_layers=periodic_layers, dirac_mode = True)
total_time_steps = 365 * day_interval  # Yup
time = 0
for i in range(total_time_steps):
    current_reward = reward_t[i % day_interval]
    current_loss = loss_t[i % day_interval]
    current_foraging = foraging_gain_builder(eco)
    payoff_matrix = total_payoff_matrix_builder_memory_improved(eco, eco.populations,
                                                                total_reward_matrix=current_reward,
                                                                total_loss_matrix=current_loss,
                                                                foraging_gain=current_foraging)

    pop_old = np.copy(eco.populations)
    population_list.append(pop_old)
    eco.parameters.layered_attack = periodic_layers[i % day_interval]

    prior_sol = lemke_optimizer(eco, payoff_matrix=payoff_matrix)
    x_res = (prior_sol[0:eco.populations.size * eco.layers]).reshape((eco.populations.size, -1))
    strategy_list.append(x_res)

    #    print(np.argmin(payoff_matrix), np.argmin(total_payoff_matrix_builder(eco)), np.min(payoff_matrix), np.min(total_payoff_matrix_builder(eco)))
    delta_pop = eco.total_growth(x_res)
    new_pop = delta_pop * time_step + eco.populations
    error = np.linalg.norm(new_pop - pop_old)

    eco.population_setter(eco.total_growth(x_res) * time_step + eco.populations)
    eco.strategy_setter(x_res)
    r_c = np.copy(eco.water.res_counts)
    resource_list.append(r_c)

    eco.water.update_resources(consumed_resources=eco.consumed_resources(), time_step=time_step)
    print("I'm here")
    print(error, eco.populations, np.sum(eco.water.res_counts), time_step, new_pop - pop_old,
          np.cos(i * 2 * np.pi / day_interval), i/total_time_steps)
    time += time_step

with open('eco_dyn_res_lemke.pkl', 'wb') as f:
    pkl.dump(eco, f, pkl.HIGHEST_PROTOCOL)

with open('strategies_eco_dyn_res_lemke.pkl', 'wb') as f:
    pkl.dump(strategy_list, f, pkl.HIGHEST_PROTOCOL)

with open('population_eco_dyn_res_lemke.pkl', 'wb') as f:
    pkl.dump(population_list, f, pkl.HIGHEST_PROTOCOL)

with open('resource_eco_dyn_res_lemke.pkl', 'wb') as f:
    pkl.dump(resource_list, f, pkl.HIGHEST_PROTOCOL)

with open('rewards_eco_dyn_res_lemke.pkl', 'wb') as f:
    pkl.dump(reward_t, f, pkl.HIGHEST_PROTOCOL)

with open('losses_eco_dyn_res_lemke.pkl', 'wb') as f:
    pkl.dump(loss_t, f, pkl.HIGHEST_PROTOCOL)

with open('periodic_layers_eco_dyn_res_lemke.pkl', 'wb') as f:
    pkl.dump(periodic_layers, f, pkl.HIGHEST_PROTOCOL)