from size_based_ecosystem import *
import copy as copy
import scipy.stats as stats
import pickle as pkl

depth = 10
layers = 100
size_classes = 1
lam = 1
simulate = True
verbose = False
daily_cycle = 365*2*np.pi


obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
#norm_dist = stats.norm.pdf(obj.x, loc = 3, scale = 3)
#print(norm_dist)
norm_dist = stats.norm.pdf(obj.x, loc = 3)
res_start = 3*norm_dist #0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)
list_of_sizes = np.array([20, 5000, 1]) #, 1, 400, 1600, 40000])

l2 = False
size_classes = 2
m_v_t = list_of_sizes[0:size_classes]
params = ecosystem_parameters(m_v_t, obj)
eco = ecosystem_optimization(m_v_t, layers, params, obj, water_start, l2 = l2, output_level = 5, movement_cost = 1)
#OG_layered_attack = np.copy(eco.parameters.layered_attack)
eco.population_setter(np.array([20, 0.01]) )#, 1, 1, 1, 0.1]))
eco.parameters.handling_times = np.array([0, 0]) * eco.parameters.handling_times

frozen_ecos = []
def one_actor_steady_state(pop, eco=eco):
    eco.populations = pop
    x_res = sequential_nash(eco, verbose=verbose)
    eco.strategy_setter(x_res)
    return eco.total_growth()+eco


stability = False
time_step = 10**(-4)
#max_err = time_step*1/10

if simulate is True:
    while size_classes < len(list_of_sizes)+1:

        x_res = sequential_nash(eco, verbose=True, l2=l2, max_its_seq = 20)
        #x_res_verify = hillclimb_nash(eco, verbose=True, l2=l2)
        #print(np.max(np.abs(x_res-x_res_verify)), "Difference")

        pop_old = np.copy(eco.populations)
        delta_pop = eco.total_growth(x_res)
        new_pop = delta_pop * time_step + eco.populations
        error = np.linalg.norm(new_pop - pop_old)

        #while error>0.01 or min(new_pop)>1.1*min(pop_old):
        #    new_pop = delta_pop * time_step + eco.populations
        #    error = np.linalg.norm(new_pop - pop_old)
        #    time_step = max(0.75*time_step, 10**(-12))
        #    eco.heat_kernel_creator(time_step)

        x_res = sequential_nash(eco, verbose=True, l2=l2, max_its_seq=20, time_step=time_step)

        eco.population_setter(eco.total_growth(x_res) * time_step + eco.populations)
        eco.strategy_setter(x_res)
#        eco.water.update_resources(consumed_resources=eco.consumed_resources(), time_step=time_step) Currently fixing.
        print("I'm here")
        print(error, eco.populations, np.sum(eco.water.res_counts), time_step, new_pop - pop_old)

        #if error<10**(-4):
        #    time_step = min(5/4*time_step, 10**(-4))
        #    eco.heat_kernel_creator(time_step)




        if error/time_step < min(1/100, np.min(eco.populations/2)):
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
            eco = ecosystem_optimization(m_v_t, layers, params, obj, water_start, l2 = l2, time_step=time_step)
            eco.strategy_matrix[0:size_classes-1] = strat_old
            eco.populations[0:size_classes-1] = pops
            eco.populations[-1] = 10**(-10)
            eco.parameters.handling_times = np.array([0, 0, 0])*eco.parameters.handling_times

            stability = False

    with open('eco_systems.pkl', 'wb') as f:
        pkl.dump(frozen_ecos, f, pkl.HIGHEST_PROTOCOL)
