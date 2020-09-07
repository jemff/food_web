from size_based_ecosystem import *
import copy as copy
import scipy.stats as stats
import pickle as pkl

depth = 20 #Previously 5 has worked well.
layers = 60 #5 works well.
size_classes = 1
lam = 2
simulate = True
verbose = False
daily_cycle = 365*2*np.pi


obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
norm_dist = stats.norm.pdf(obj.x, loc = 2)
res_start = 3*norm_dist #0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)
list_of_sizes = np.array([20, 8000]) #, 1, 400, 1600, 40000])

l2 = False
size_classes = 1
m_v_t = np.array([list_of_sizes[size_classes - 1]])
params = ecosystem_parameters(m_v_t, obj)
eco = ecosystem_optimization(m_v_t, layers, params, obj, water_start, l2 = l2)
#OG_layered_attack = np.copy(eco.parameters.layered_attack)
eco.population_setter(np.array([1]) )#, 1, 1, 1, 0.1]))
eco.parameters.handling_times = np.array([0])

frozen_ecos = []
def one_actor_steady_state(pop, eco=eco):
    eco.populations = pop
    x_res = sequential_nash(eco, verbose=verbose)
    eco.strategy_setter(x_res)
    return eco.total_growth()+eco


stability = False
time_step = 10**(-7)
#max_err = time_step*1/10

if simulate is True:
    while size_classes < len(list_of_sizes)+1:

        x_res = sequential_nash(eco, verbose=verbose)
        eco.strategy_setter(x_res)
        pop_old = np.copy(eco.populations)
        eco.population_setter(eco.total_growth() * time_step + eco.populations)
        error = np.linalg.norm(eco.populations - pop_old)
        print(error, eco.populations, np.sum(eco.water.res_counts), time_step)
        eco.water.update_resources(consumed_resources = eco.consumed_resources(), time_step = time_step)

        if error>0.01:
            time_step = max(0.75*time_step, 10**(-12))
        else:
            time_step = min(5/4*time_step, 10**(-7))



        if error/time_step < min(1/10, np.min(eco.populations/2)):
            stability = True

        if stability is True:
            old_eco = copy.deepcopy(eco)
            strat_old = np.copy(eco.strategy_matrix)

            print("New regime")
            frozen_ecos.append(old_eco)
            pops = np.copy(old_eco.populations)
            size_classes += 1
            m_v_t = np.copy(list_of_sizes[0:size_classes])

            params = ecosystem_parameters(m_v_t, obj)
            print(params.forager_or_not)
            eco = ecosystem_optimization(m_v_t, layers, params, obj, water_start, l2 = l2)
            eco.strategy_matrix[0:size_classes-1] = strat_old
            eco.populations[0:size_classes-1] = pops
            eco.populations[-1] = 10**(-10)
            eco.parameters.handling_times = np.array([0, 0])

            stability = False

    with open('eco_systems.pkl', 'wb') as f:
        pkl.dump(frozen_ecos, f, pkl.HIGHEST_PROTOCOL)
