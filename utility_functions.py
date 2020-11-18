import siconos.numerics as sn
import pickle as pkl
import copy as copy
from size_based_ecosystem import *

import pandas as pd
import pvlib
from pvlib import clearsky

def heat_kernel(spectral, t, k):
    gridx, gridy = np.meshgrid(spectral.x, spectral.x)
    ker = lambda x, y: np.exp(-(x - y) ** 2 / (4 * k * t)) + np.exp(-(-y - x) ** 2 / (4 * k * t)) + np.exp(-(2*spectral.x[-1] - x - y) ** 2 / (4 * k * t))
    out = (4 * t * k * np.pi) ** (-1 / 2) * ker(gridx, gridy)
    normalizations = np.sum(spectral.M @ out, axis = 0)
    normalizations = np.diag(1/normalizations)
    return normalizations @ spectral.M @ out

def total_payoff_matrix_builder_sparse(eco, current_layered_attack = None, dirac_mode = False):
    total_payoff_matrix = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))

    if current_layered_attack is None:
        current_layered_attack = eco.parameters.layered_attack

    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            if i != j:
                i_vs_j = jit_wrapper(eco, i, j, current_layered_attack = current_layered_attack, dirac_mode = dirac_mode) #payoff_matrix_builder(eco, i, j, current_layered_attack = current_layered_attack, dirac_mode = dirac_mode)
            elif i == j:
                i_vs_j = np.zeros((eco.layers, eco.layers))
            #if i == 1:
            #    total_payoff_matrix[i*eco.layers:(i+1)*eco.layers, j*eco.layers: (j+1)*eco.layers] = i_vs_j.T
            #else:

            total_payoff_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = i_vs_j
#    print("MAXIMM PAYDAY ORIGINAL",  np.max(total_payoff_matrix))
    total_payoff_matrix[total_payoff_matrix != 0] = total_payoff_matrix[total_payoff_matrix != 0] - np.max(total_payoff_matrix) - 1 #Making sure everything is negative  #- 0.00001
    #total_payoff_matrix = total_payoff_matrix/np.max(-total_payoff_matrix)
    #print(np.where(total_payoff_matrix == 0))
    return total_payoff_matrix

def lemke_optimizer(eco, payoff_matrix = None, dirac_mode = True):
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

    z = np.zeros((eco.layers*eco.populations.size+eco.populations.size,), np.float64)
    w = np.zeros_like(z)
    options = sn.SolverOptions(sn.SICONOS_LCP_PIVOT)
    #sn.SICONOS_IPARAM_MAX_ITER = 10000000<
    options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = 1000000
    options.dparam[sn.SICONOS_DPARAM_TOL] = 10**(-5)
    info = sn.linearComplementarity_driver(lcp, z, w, options)
    if sn.lcp_compute_error(lcp,z,w, ztol) > 10**(-5):
     print(sn.lcp_compute_error(lcp,z,w, ztol), "Error")
    return z


def population_growth(eco, populations, resources, payoff_matrix = None, lemke = True, time_step = 1/(96*365)):
    eco_temp = copy.deepcopy(eco)
    eco_temp.populations = populations
    eco_temp.water.res_counts = resources
    if payoff_matrix is None:
        payoff_matrix = total_payoff_matrix_builder(eco)
    if lemke is True:
        prior_sol = lemke_optimizer(eco, payoff_matrix=payoff_matrix)
    else:
        prior_sol = quadratic_optimizer(eco, payoff_matrix=payoff_matrix)
    x_res = (prior_sol[0:eco.populations.size * eco.layers]).reshape((eco.populations.size, -1))


    delta_pop = eco.total_growth(x_res)

    eco_temp.population_setter(delta_pop * time_step + eco.populations)
    eco_temp.strategy_setter(x_res)
    eco_temp.water.update_resources(consumed_resources=eco.consumed_resources(), time_step=time_step)

    return eco.populations, eco.water.res_counts, x_res

def fix_point_finder(eco, day_interval, x0, resources0, min_attack_rate, darkness_length):
    error = 1
    populations = np.copy(x0)
    resources = np.copy(resources0)
    i = 0
    periodic_layers = periodic_attack(eco.parameters.layered_attack, day_interval=day_interval, minimum_attack=min_attack_rate,
                                      darkness_length= darkness_length)
    reward_t, loss_t = reward_loss_time_dependent(eco, periodic_layers=periodic_layers)

    while error > 10**(-6):
        if i is 0:
            pops_start, res_start, strat_start = population_growth(eco, populations, resources)
        if i == day_interval-1:
            pops_end, res_end, strat_end = population_growth(eco, populations, resources)
            error = np.linalg.norm(pops_end-pops_start)

        current_foraging = foraging_gain_builder(eco, resources)
        current_reward = reward_t[i % day_interval]
        current_loss = loss_t[i % day_interval]

        payoff_matrix = total_payoff_matrix_builder_memory_improved(eco, populations,
                                                                    total_reward_matrix=current_reward,
                                                                    total_loss_matrix=current_loss,
                                                                    foraging_gain=current_foraging)

        populations, resources, x_res = population_growth(eco, populations, resources, payoff_matrix=payoff_matrix)
        if error > 10**(-6):
            i = (i + 1) % day_interval



def simulator(eco, params, filename, h_k = None, lemke = True, min_attack_rate = 10**(-4), total_days = 365, day_interval = 96, darkness_length = 2):
    population_list = []
    resource_list = []
    strategy_list = []

    time_step = (1 / 365) * (1 / day_interval)

    if h_k is None:
        h_k = heat_kernel(eco.spectral, time_step, 90000)

    periodic_layers = periodic_attack(params.layered_attack, day_interval=day_interval, minimum_attack=min_attack_rate,
                                      darkness_length= darkness_length)

    reward_t, loss_t = reward_loss_time_dependent(eco, periodic_layers=periodic_layers)
    total_time_steps = int(total_days * day_interval)  # Yup
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
        if lemke is True:
            prior_sol = lemke_optimizer(eco, payoff_matrix=payoff_matrix)
        else:
            prior_sol = quadratic_optimizer(eco, payoff_matrix=payoff_matrix)
        x_res = (prior_sol[0:eco.populations.size * eco.layers]).reshape((eco.populations.size, -1))
        strategy_list.append(x_res)

        delta_pop = eco.total_growth(x_res)
        new_pop = delta_pop * time_step + eco.populations
        error = np.linalg.norm(new_pop - pop_old)

        eco.population_setter(eco.total_growth(x_res) * time_step + eco.populations)
        eco.strategy_setter(x_res)
        r_c = np.copy(eco.water.res_counts)
        resource_list.append(r_c)

        eco.water.update_resources(consumed_resources=eco.consumed_resources(), time_step=time_step)
        eco.water.res_counts = eco.water.res_counts @ h_k

        print(error, eco.populations, np.sum(eco.water.res_counts), time_step, new_pop - pop_old,
              np.cos(i * 2 * np.pi / day_interval), i / total_time_steps)
        time += time_step

    with open('eco'+filename+'.pkl', 'wb') as f:
        pkl.dump(eco, f, pkl.HIGHEST_PROTOCOL)

    with open('strategies' + filename + '.pkl', 'wb') as f:
        pkl.dump(strategy_list, f, pkl.HIGHEST_PROTOCOL)

    with open('population' + filename + '.pkl', 'wb') as f:
        pkl.dump(population_list, f, pkl.HIGHEST_PROTOCOL)

    with open('resource' + filename + '.pkl', 'wb') as f:
        pkl.dump(resource_list, f, pkl.HIGHEST_PROTOCOL)

    with open('rewards' + filename + '.pkl', 'wb') as f:
        pkl.dump(reward_t, f, pkl.HIGHEST_PROTOCOL)

    with open('losses'  + filename + '.pkl', 'wb') as f:
        pkl.dump(loss_t, f, pkl.HIGHEST_PROTOCOL)

    with open('periodic'  + filename + '.pkl', 'wb') as f:
        pkl.dump(periodic_layers, f, pkl.HIGHEST_PROTOCOL)


def phyto_growth(t, lat, depth):
    """t in days, latitude in degrees, depth in meters"""

    return np.exp(-0.025 * depth) * (1 - 0.8 * np.sin(np.pi * lat / 180) * np.cos(2 * np.pi * t / 365))

def attack_coefficient(It, z, k=0.05*2, beta_0 = 10**(-4)):
    """It in watt, z is a vector of depths in meters, k in m^{-1}"""
    return 2*It*np.exp(-k*z)/(1+It*np.exp(-k*z))+beta_0

def new_layer_attack(params, solar_levels, k = 0.05*2, beta_0 = 10**(-4)):
    weights = attack_coefficient(solar_levels, params.spectral.x, k = k, beta_0=beta_0)
    layers = np.zeros((params.spectral.x.shape[0], *params.attack_matrix.shape))

    for i in range(params.spectral.x.shape[0]):
        layers[i] = weights[i] * params.attack_matrix


    return layers


#latitude = 34.5531, longitude = 18.0480 #Middelhavet
def solar_input_calculator(latitude = 55.571831046, longitude = 12.822830042, tz = 'Europe/Vatican', name = 'Oresund', start_date = '2014-04-01', end_date = '2014-10-01', freq = '15Min', normalized = True):
    altitude = 0
    times = pd.date_range(start=start_date, end=end_date, freq=freq, tz=tz)

    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    apparent_elevation = solpos['apparent_elevation']

    aod700 = 0.1

    precipitable_water = 1

    pressure = pvlib.atmosphere.alt2pres(altitude)

    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    # an input is a Series, so solis is a DataFrame

    solis = clearsky.simplified_solis(apparent_elevation, aod700, precipitable_water,

                                      pressure, dni_extra)

    if normalized is True:
        return solis.dhi.values/np.max(solis.dhi.values)
    else:
        return solis.dhi.values



def simulator_new(eco, filename, h_k = None, lemke = True, min_attack_rate = 10**(-4), start_date =  '2014-04-01',
                  end_date = '2014-10-01', day_interval = 96, latitude = 55.571831046, longitude = 12.822830042,
                  optimal=True, diffusion = 5000, k = 4*0.05, sparse = True, population_dynamics = True):
    population_list = []
    resource_list = []
    strategy_list = []

    time_step = (1 / 365) * (1 / day_interval)
    solar_levels = solar_input_calculator(latitude=latitude, longitude=longitude, start_date=start_date, end_date = end_date)
    print(len(solar_levels))
    if h_k is None:
        h_k = heat_kernel(eco.spectral, time_step, diffusion)

    total_time_steps = len(solar_levels)
    time = 0
    for i in range(total_time_steps):
        current_layered_attack = new_layer_attack(eco.parameters, solar_levels[i], beta_0=min_attack_rate, k = k)
        pop_old = np.copy(eco.populations)
        population_list.append(pop_old)
        if optimal is True:
            if sparse is False:
                payoff_matrix = total_payoff_matrix_builder(eco, current_layered_attack)
            else:
                payoff_matrix = total_payoff_matrix_builder_sparse(eco, current_layered_attack)
            if lemke is True:
                prior_sol = lemke_optimizer(eco, payoff_matrix=payoff_matrix)
            else:
                prior_sol = quadratic_optimizer(eco, payoff_matrix=payoff_matrix)
        if optimal is False:
            prior_sol = np.copy(eco.strategy_matrix)
        x_res = (prior_sol[0:eco.populations.size * eco.layers]).reshape((eco.populations.size, -1))
        strategy_list.append(x_res)

        delta_pop = eco.total_growth(x_res)
        eco.parameters.layered_attack = current_layered_attack
        new_pop = delta_pop * time_step + eco.populations
        error = np.linalg.norm(new_pop - pop_old)
        if population_dynamics is True:
            eco.population_setter(eco.total_growth(x_res) * time_step + eco.populations)
        eco.strategy_setter(x_res)
        r_c = np.copy(eco.water.res_counts)
        resource_list.append(r_c)
        if population_dynamics is True:
            eco.water.update_resources(consumed_resources=eco.consumed_resources(), time_step=time_step)
            eco.water.res_counts = eco.water.res_counts @ h_k

        print(error, eco.populations, np.sum(eco.water.res_counts), time_step, new_pop - pop_old, solar_levels[i])
        time += time_step

    with open('eco'+filename+'.pkl', 'wb') as f:
        pkl.dump(eco, f, pkl.HIGHEST_PROTOCOL)

    with open('strategies' + filename + '.pkl', 'wb') as f:
        pkl.dump(strategy_list, f, pkl.HIGHEST_PROTOCOL)

    with open('population' + filename + '.pkl', 'wb') as f:
        pkl.dump(population_list, f, pkl.HIGHEST_PROTOCOL)

    with open('resource' + filename + '.pkl', 'wb') as f:
        pkl.dump(resource_list, f, pkl.HIGHEST_PROTOCOL)