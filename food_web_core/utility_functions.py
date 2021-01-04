import siconos.numerics as sn
import pickle as pkl
import copy as copy
#from size_based_ecosystem import *
import numpy as np
from scipy import special
import pandas as pd
import pvlib
from pvlib import clearsky
from numba import njit



def sequential_nash(eco, verbose = False, l2 = False, max_its_seq = None, time_step = 10**(-4), max_mem = None):
    x_temp = np.copy(eco.strategy_matrix)
    smooth_xtemp = np.copy(x_temp)
    x_temp2 = np.copy(eco.strategy_matrix)
    smooth_xtemp2 = np.copy(x_temp2)
    error = 1
    iterations = 0

    if max_mem is None:
        max_mem = eco.populations.shape[0]**eco.populations.shape[0]+eco.populations.shape[0]
    if max_its_seq is None:
        max_its_seq = 4*max_mem

    memory_keeper = np.zeros((max_mem, eco.strategy_matrix.shape[0], eco.strategy_matrix.shape[1]))
    mem = 0
    while error > 10 ** (-6) and iterations <= max_its_seq:
        for k in range(eco.mass_vector.shape[0]):
            result = eco.one_actor_growth(k, x_temp_i=x_temp, time_step=time_step)
            x_temp2[k] = np.array(result).flatten()
            if verbose is True:
                print(np.sum(np.dot(eco.ones, np.dot(eco.spectral.M, result))), k, "1 Norm of strategy")
                print(np.sum(np.dot(eco.ones, np.dot(eco.spectral.M, smooth_xtemp[k]))), k, "1 Norm of old smooth strategy")

        if l2 is False:
            for k in range(eco.populations.shape[0]):
                smooth_xtemp[k] = x_temp[k] @ eco.heat_kernels[k]  # Going smooth.
                smooth_xtemp2[k] = x_temp2[k] @ eco.heat_kernels[k]  # Going smooth.

            errors = np.sum(np.dot(np.abs(smooth_xtemp2 - smooth_xtemp), eco.spectral.M), axis = 1)
            error = np.max(errors)
            print(errors.shape)

        else:
            print(np.dot((x_temp2 - x_temp), eco.spectral.M).shape)
            error = np.max(np.dot((smooth_xtemp - smooth_xtemp2), eco.spectral.M) @ (x_temp2 - x_temp).T)

        if verbose is True:
            print("Error: ", error, errors)
        good_mem = 0
        mem_err = 1
        if error > 10**(-6):
            for k in range(memory_keeper.shape[0]):
                mem_err = np.sum(np.abs(memory_keeper[k] - x_temp2))
                if mem_err < 10**(-6):
                    good_mem = k

        if mem_err<10**(-6):
            bottom = min(mem, good_mem)
            top = max(mem, good_mem)
            if top == bottom:
                top = max_mem
                bottom = 0
            else:
                top += 1
            x_temp = np.mean(memory_keeper[bottom: top], axis = 0)
            print(x_temp, "Memory used for mean", memory_keeper[bottom: top], "Total memory",  memory_keeper)
            print("Memory working", top, bottom)
            error = mem_err
        else:
            x_temp = np.copy(x_temp2)

         #(error_col) * x_temp2 + (1-1/error_col)*x_temp #np.copy(x_temp2)
        memory_keeper[mem] = np.copy(x_temp2)
        mem += 1
        mem = int(mem % max_mem)
        #print(mem)
        iterations += 1
    iterations_newt = 0
    if iterations > max_its_seq and error > 10**(-6):
        print("Entering newton-phase for strategy mixing")


    return x_temp


#def lemke_optimizer(eco, payoff_matrix = None):
#    A = np.zeros((eco.populations.size, eco.populations.size * eco.layers))
#    for k in range(eco.populations.size):
#        A[k, k * eco.layers:(k + 1) * eco.layers] = -1

#    q = np.zeros(eco.populations.size + eco.populations.size * eco.layers)
#    q[eco.populations.size * eco.layers:] = -1
#    q = q.reshape(-1, 1)
#    if payoff_matrix is None:
#        payoff_matrix = total_payoff_matrix_builder(eco)

#    H = np.block([[-payoff_matrix, A.T], [-A, np.zeros((A.shape[0], eco.populations.size))]])

#    sol = lcp.lemkelcp(H, q, maxIter = 10000)
#    if sol[1]==0:
#        return sol[0][0:eco.layers*eco.populations.size].reshape((eco.populations.size, -1))
#    else:
#        return quadratic_optimizer(eco, payoff_matrix=payoff_matrix)[0:eco.layers*eco.populations.size].reshape((eco.populations.size, -1))

def quadratic_optimizer(eco, payoff_matrix = None, prior_sol=None):

    A=np.zeros((eco.populations.size, eco.populations.size*eco.layers))
    if eco.spectral.segments == 1:
        for k in range(eco.populations.size):
            A[k,k*eco.layers:(k+1)*eco.layers] = -1
    if eco.spectral.segments != 1:
        Temp = -np.copy(eco.ones).astype(float)
        Temp[::eco.spectral.n] += 1/2
        Temp[eco.spectral.n-1::eco.spectral.n] += 1/2

        Temp[0] = -1
        Temp[-1] = -1
        print(Temp)
        for k in range(eco.populations.size):
            A[k, k * eco.layers:(k + 1) * eco.layers] = -Temp

    for k in range(eco.populations.size):
        A[k, k * eco.layers:(k + 1) * eco.layers] = -1

    q = np.zeros(eco.populations.size+eco.populations.size*eco.layers)
    q[eco.populations.size*eco.layers:] = -1
    q = q.reshape(-1, 1)
    if payoff_matrix is None:
        payoff_matrix = total_payoff_matrix_builder(eco)

    p = ca.SX.sym('p', eco.populations.size*eco.layers)
    y = ca.SX.sym('y', eco.populations.size)
    u = ca.SX.sym('u', eco.populations.size*eco.layers)
    v = ca.SX.sym('v', eco.populations.size)

    cont_conds = []
    if eco.spectral.segments>1:
        for j in range(eco.spectral.segments-1):
            cont_conds.append(p[j*eco.spectral.n] - p[(j+1)*eco.spectral.n])
            cont_conds.append(u[j*eco.spectral.n] - u[(j+1)*eco.spectral.n])

    print("Here")
    z = ca.vertcat(*[p, y])
    w = ca.vertcat(*[u, v])
    #
#    print(np.block([-A]).shape)
    H = np.block([[-payoff_matrix, A.T], [-A, np.zeros((A.shape[0], eco.populations.size))]])
  #  w = H @ z + q

#    print(H.shape)
    print("Here")
    f = w.T @ z #ca.norm_2()
    if eco.spectral.segments > 1:
        g = ca.vertcat(*[*cont_conds, w - H @ z - q])

    else:
        g = w - H @ z - q #ca.norm_2() H @ z + q

    print("Here")

    x = ca.vertcat(z, w)
    lbx = np.zeros(x.size())+10**(-8)
    ubg = np.zeros(g.size()) #[ca.inf]*int(g.size()[0]) #np.zeros(g.size())
    lbg = np.zeros(g.size())

    print("Just before optimizing")
    s_opts = {'ipopt': {'print_level': 5, 'tol':1E-8}}
    prob = {'x': x, 'f': f, 'g': g}
    solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
    print("Solver decleared")
    #prior_sol = False
    if prior_sol is None:
        sol = solver(lbx=lbx, lbg=lbg, ubg = ubg) #ubg=ubg
        print("Solved")

    else:
        sol = solver(x0=prior_sol, lbx=lbx, lbg=lbg, ubg = ubg) #ubg=ubg,


    print(sol['f'])
    x_out = np.array(sol['x']).flatten()
    #print(np.min(x_out), np.dot(x_out[0:eco.populations.size*(eco.layers+1)], x_out[eco.populations.size*(eco.layers+1):]))
    #print(x_out[0:eco.layers*eco.populations.size])
    return x_out





def total_payoff_matrix_builder(eco, current_layered_attack = None, dirac_mode = False):
    total_payoff_matrix = np.zeros((eco.populations.size*eco.spectral.n*eco.spectral.segments, eco.populations.size*eco.spectral.n*eco.spectral.segments))

    if current_layered_attack is None:
        current_layered_attack = eco.parameters.layered_attack

    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            if i != j:
                i_vs_j = jit_wrapper(eco, i, j, current_layered_attack = current_layered_attack, dirac_mode = dirac_mode) #payoff_matrix_builder
            elif i == j:
                i_vs_j = np.zeros((eco.layers, eco.layers))
            #if i == 1:
            #    total_payoff_matrix[i*eco.layers:(i+1)*eco.layers, j*eco.layers: (j+1)*eco.layers] = i_vs_j.T
            #else:

            total_payoff_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = i_vs_j
#    print("MAXIMM PAYDAY ORIGINAL",  np.max(total_payoff_matrix))
    total_payoff_matrix = total_payoff_matrix - np.max(total_payoff_matrix) #- 1 #Making sure everything is negative  #- 0.00001
    #total_payoff_matrix = total_payoff_matrix/np.max(-total_payoff_matrix)
    return total_payoff_matrix


#@njit(parallel=True)
def payoff_matrix_builder(eco, i, j, current_layered_attack, dirac_mode = False):
    payoff_i = np.zeros((current_layered_attack.shape[0], current_layered_attack.shape[0]))
    diracs = eco.dirac_delta_creator_i(0, normalize = True)
    for k in range(current_layered_attack.shape[0]):
        one_k_vec = np.zeros(eco.layers)
        if dirac_mode is True:
            one_k_vec = diracs[i] #np.zeros(eco.layers)
        else:
            one_k_vec[k] = 1
        for n in range(eco.layers):
            one_n_vec = np.zeros(eco.layers)
            if dirac_mode is True:
                one_n_vec = diracs[n]
            else:
                one_n_vec[n] = 1
            strat_mat = np.vstack([one_k_vec, one_n_vec])
            payoff_i[k, n] = eco.lin_growth(i, j, strat_mat, current_layered_attack)

    return payoff_i




def graph_builder_old(eco):  # Move into ecosystem class
    strat_mat = eco.strategy_matrix

    outflows = np.zeros((eco.mass_vector.shape[0]+1, eco.mass_vector.shape[0]))
    for i in range(eco.mass_vector.shape[0]):
        for k in range(eco.parameters.who_eats_who.shape[0]):
            interaction_term = eco.parameters.who_eats_who[k] * eco.populations * eco.parameters.handling_times[k] * \
                               eco.parameters.clearance_rate[k]
            layer_action = strat_mat[i, :].reshape((eco.layers, 1)) * eco.parameters.layered_attack[:, i, :] * (
                        interaction_term.reshape((eco.populations.shape[0], 1)) * strat_mat).T

            foraging_term = eco.water.res_counts * eco.parameters.forager_or_not[k] * \
                            eco.parameters.handling_times[k] \
                            * eco.parameters.clearance_rate[k] * eco.parameters.layered_foraging[:, k]
            # print(eco.parameters.layered_attack[:,k,i], k, i, strat_mat[i])
            outflows[1+i, k] = np.dot(strat_mat[k], np.dot(eco.spectral.M, strat_mat[i] * eco.populations[
                k] * eco.parameters.layered_attack[:, k, i] * eco.parameters.clearance_rate[k] * eco.populations[
                                                             i])) / \
                             (1 + np.sum(np.dot(eco.ones, np.dot(eco.spectral.M,
                                                                 np.sum(layer_action, axis=1) + foraging_term))))
            outflows[0,k] =  np.sum(np.dot(eco.spectral.M, foraging_term)/(1 + np.sum(np.dot(eco.ones, np.dot(eco.spectral.M,
                                                                 np.sum(layer_action, axis=1) + foraging_term)))))
    return outflows


def graph_builder(eco, layered_attack=None, populations=None, resources=None, strategies=None):  # Move into ecosystem class
    if layered_attack is None:
        layered_attack = eco.parameters.layered_attack
    if populations is None:
        populations = eco.populations
    if resources is None:
        resources = eco.water.res_counts
    if strategies is None:
        strategies = np.copy(eco.strategy_matrix)

    classes = populations.size
    inflows = np.zeros((classes + 1, classes + 1))
    x_temp = np.zeros((2, eco.layers))

    for i in range(classes):
        x_temp[0] = strategies[i] @ eco.heat_kernels[0]  # Going smooth.
        inflows[i + 1, 0] = (eco.parameters.forager_or_not[i] * eco.parameters.clearance_rate[
            i] * eco.parameters.layered_foraging[:, i] * resources) @ (eco.spectral.M @ x_temp[0])

        for j in range(classes):
            x_temp[1] = strategies[j] @ eco.heat_kernels[0]  # Going smooth.
            x = x_temp[0].reshape(-1, 1)
            interaction_term = eco.parameters.who_eats_who[i, j] * eco.parameters.clearance_rate[i]
            lin_growth = interaction_term * (x_temp[1] * layered_attack[:, i, j].T) @ (eco.spectral.M @ x)

            actual_growth = eco.parameters.efficiency * lin_growth

            inflows[i + 1, j + 1] = actual_growth * populations[j]

    return inflows


def periodic_attack(layered_attack, day_interval = 96, darkness_length = 0, minimum_attack = 0):
    OG_layered_attack = np.copy(layered_attack)
    periodic_layers = np.zeros((day_interval,*layered_attack.shape))
    for i in range(day_interval):
        periodic_layers[i] = 1/2*(1+minimum_attack+min(max((darkness_length+1)*np.cos(i*2*np.pi/day_interval), -1), 1))*OG_layered_attack

    return periodic_layers

def reward_loss_time_dependent(eco, periodic_layers, dirac_mode = False):
    rewards_t = np.zeros((periodic_layers.shape[0], eco.layers*eco.populations.size, eco.layers*eco.populations.size))
    losses_t = np.zeros((periodic_layers.shape[0], eco.layers*eco.populations.size, eco.layers*eco.populations.size))
    for i in range(periodic_layers.shape[0]):
        print(i)
        reward_i, loss_i = loss_and_reward_builder(eco, periodic_layers[i], dirac_mode = dirac_mode)
        rewards_t[i] = reward_i
        losses_t[i] = loss_i

    return rewards_t, losses_t


def total_payoff_matrix_builder_memory_improved(eco, populations, total_reward_matrix, total_loss_matrix, foraging_gain):
    total_rew_mat = np.copy(total_reward_matrix)
    total_loss_mat = np.copy(total_loss_matrix)

    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            total_rew_mat[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = populations[j]*total_reward_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers]
            total_loss_mat[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = populations[j]*total_loss_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers]

    total_payoff_matrix = eco.parameters.efficiency*(total_rew_mat + foraging_gain) - total_loss_mat
    #print(np.max(total_payoff_matrix), "MAXIMUM PAYDAY")
    return total_payoff_matrix - np.max(total_payoff_matrix) #- 0.00001

def foraging_gain_builder(eco, resources = None, dirac_mode = False):
    if resources is None:
        resources = eco.water.res_counts

    foragers = np.where(eco.parameters.forager_or_not == 1)
    foraging_gain = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))
    foraging_gain_i = np.zeros((eco.layers, eco.layers))
    diracs = eco.dirac_delta_creator_i(0, normalize=False)

    for forager in foragers:
        for i in range(eco.layers):
            one = np.zeros(eco.layers)
            if dirac_mode is True:
                one = diracs[i]  # np.zeros(eco.layers)
            else:
                one[i] = 1
            foraging_gain_i[i] = eco.parameters.clearance_rate[forager[0]]*(one @ eco.heat_kernels[forager[0]] * eco.parameters.layered_foraging[:,forager[0]]) @ (eco.spectral.M @ (resources))/np.sum(eco.parameters.who_eats_who[:, forager[0]])

        eaters = np.where(eco.parameters.who_eats_who[:,forager[0]] == 1)
        for eater in eaters:
            foraging_gain[forager[0] * eco.layers:(forager[0] + 1) * eco.layers, eater[0] * eco.layers: (eater[0] + 1) * eco.layers] = foraging_gain_i

    return foraging_gain



def depth_based_loss(who_eats_who, depth_loss_matrix, heat_kernels):
    layers = depth_loss_matrix.shape[1]
    actors = depth_loss_matrix.shape[0]
    giant_loss_matrix = np.zeros((layers*actors, layers*actors))
    for i in range(depth_loss_matrix.shape[0]):
        eaters = np.where(who_eats_who[:,i] == 1)
        for eater in eaters:
            giant_loss_matrix[i * layers:(i + 1) * layers, eater[0] * layers: (eater[0] + 1) * layers] = heat_kernels[i]@depth_loss_matrix[i]/np.sum(who_eats_who[:, i])

    return giant_loss_matrix


def loss_and_reward_builder(eco, layered_attack = None, dirac_mode = False):
    if layered_attack is None:
        layered_attack = eco.parameters.layered_attack

    total_reward_matrix = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))
    total_loss_matrix = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))
    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            if i != j:
                i_vs_j = reward_matrix_builder(eco, layered_attack, i, j, dirac_mode = dirac_mode)
                j_vs_i = loss_matrix_builder(eco, layered_attack, i, j, dirac_mode = dirac_mode)
            elif i == j:
                i_vs_j = np.zeros((eco.layers, eco.layers))
                j_vs_i = np.zeros((eco.layers, eco.layers))

            total_reward_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = i_vs_j
            total_loss_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = j_vs_i

    return total_reward_matrix, total_loss_matrix


def reward_matrix_builder(eco, layered_attack, i, j, dirac_mode = False):
    reward_i = np.zeros((eco.layers, eco.layers))
    diracs = eco.dirac_delta_creator_i(0, normalize = True)

    for k in range(eco.layers):
        one_k_vec = np.zeros(eco.layers)
        if dirac_mode is True:
            one_k_vec = diracs[i] #np.zeros(eco.layers)
        else:
            one_k_vec[k] = 1
        for n in range(eco.layers):
            one_n_vec = np.zeros(eco.layers)
            if dirac_mode is True:
                one_n_vec = diracs[n]
            else:
                one_n_vec[n] = 1
            strat_mat = np.vstack([one_k_vec, one_n_vec])
            reward_i[k, n] = lin_growth_no_pops_no_res(eco, i, j, layered_attack, strat_mat)

    return reward_i


def loss_matrix_builder(eco, layered_attack, i, j, dirac_mode = False):
    loss_i = np.zeros((eco.layers, eco.layers))
    diracs = eco.dirac_delta_creator_i(0, normalize = True)
    for k in range(eco.layers):
        one_k_vec = np.zeros(eco.layers)
        if dirac_mode is True:
            one_k_vec = diracs[i] #np.zeros(eco.layers)
        else:
            one_k_vec[k] = 1
        for n in range(eco.layers):
            one_n_vec = np.zeros(eco.layers)
            if dirac_mode is True:
                one_n_vec = diracs[n]
            else:
                one_n_vec[n] = 1
            strat_mat = np.vstack([one_k_vec, one_n_vec])
            loss_i[k, n] = lin_growth_no_pops_no_res(eco, j, i, layered_attack, strat_mat)

    return loss_i


def heat_kernel(spectral, t, k):
    gridx, gridy = np.meshgrid(spectral.x, spectral.x)
    ker = lambda x, y: np.exp(-(x - y) ** 2 / (4 * k * t)) + np.exp(-(-y - x) ** 2 / (4 * k * t)) + np.exp(-(2*spectral.x[-1] - x - y) ** 2 / (4 * k * t))
    out = (4 * t * k * np.pi) ** (-1 / 2) * ker(gridx, gridy)
    normalizations = np.sum(spectral.M @ out, axis = 0)
    normalizations = np.diag(1/normalizations)
    return normalizations @ spectral.M @ out

def total_payoff_matrix_builder_sparse(eco, current_layered_attack = None, dirac_mode = False, depth_based_loss = None):
    total_payoff_matrix = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))

    if current_layered_attack is None:
        current_layered_attack = eco.parameters.layered_attack
    if depth_based_loss is None:
        depth_based_loss = np.copy(total_payoff_matrix)

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
    total_payoff_matrix = total_payoff_matrix - depth_based_loss
    total_payoff_matrix[total_payoff_matrix != 0] = total_payoff_matrix[total_payoff_matrix != 0] - np.max(total_payoff_matrix) - 1 #Making sure everything is negative  #- 0.00001

    #total_payoff_matrix = total_payoff_matrix/(np.max(total_payoff_matrix)-np.min(total_payoff_matrix))
    #total_payoff_matrix[total_payoff_matrix != 0] = total_payoff_matrix[total_payoff_matrix != 0] - 0.01
    #print(np.where(total_payoff_matrix == 0))
    return total_payoff_matrix

def lemke_optimizer(eco, payoff_matrix = None, return_all = False, dirac_mode = True):
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
    #sn.SICONOS_IPARAM_MAX_ITER = 10000000
    options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = 1000000
    options.dparam[sn.SICONOS_DPARAM_TOL] = 10**(-5)
    info = sn.linearComplementarity_driver(lcp, z, w, options)
    if sn.lcp_compute_error(lcp,z,w, ztol) > 10**(-5):
     print(sn.lcp_compute_error(lcp, z, w, ztol), "Error")

    #w_ret, z_ret, ret = lcpp.LCPSolve(H, q.flatten(), pivtol=10**(-8))

    #print(ret, np.linalg.norm(z_ret-z))
    if return_all is False:
        return z
    else:
        return z, w

def phyto_growth(t, lat, depth):
    """t in days, latitude in degrees, depth in meters"""

    return np.exp(-0.025 * depth) * (1 - 0.8 * np.sin(np.pi * lat / 180) * np.cos(2 * np.pi * t / 365))

def attack_coefficient(It, z, k=0.05*2, beta_0 = 10**(-4)):
    """It in watt, z is a vector of depths in meters, k in m^{-1}"""
    return 2*It*np.exp(-k*z)/(1+It*np.exp(-k*z))+beta_0

def depth_dependent_clearance(I0, z, swim_speed=8, beta_0=10**(-2), k=0.1, c=10**(6), K = 10**(3)): #The swmming speed is encoded in beta elsewhere in the code, this way ensures maximum flexibility...?
    """ k specifies the attenueation rate, c is the light detection threshold, I specifies the light level"""

    I = I0*np.exp(-k*z)
    D = np.real(2.0 * special.lambertw(np.sqrt((K + I) * I / c) * k / 2.0) / k)

    beta = swim_speed*D**2+swim_speed*beta_0**2

    return beta

def new_layer_attack(params, solar_levels, k = 0.05*2, beta_0 = 10**(-4)):
    weights = attack_coefficient(solar_levels, params.spectral.x, k = k, beta_0=beta_0) #attack_coefficient(solar_levels, params.spectral.x, k = k, beta_0=beta_0)
    layers = np.zeros((params.spectral.x.shape[0], *params.attack_matrix.shape))

    for i in range(params.spectral.x.shape[0]):
        layers[i] = weights[i] * params.attack_matrix


    return layers

def layer_attack_physiological(params, solar_levels, k = 0.05*2, beta_0 = 10**(-4)):
    weights = attack_coefficient(solar_levels, params.spectral.x, k = k, beta_0=beta_0) #attack_coefficient(solar_levels, params.spectral.x, k = k, beta_0=beta_0)
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


def jit_wrapper(eco, i, j, current_layered_attack, dirac_mode = False):
    resources = eco.water.res_counts
    populations = eco.populations
    current_layered_attack = current_layered_attack
    heat_kernel = eco.heat_kernels[0]
    clearance_rate = eco.parameters.clearance_rate
    M = eco.spectral.M
    who_eats_who = eco.parameters.who_eats_who
    forager_or_not = eco.parameters.forager_or_not
    layered_foraging = eco.parameters.layered_foraging
    efficiency = eco.parameters.efficiency
    payoff_i = jit_payoff_matrix_builder(resources, populations, i, j, current_layered_attack = current_layered_attack,
                   heat_kernel = heat_kernel, clearance_rate = clearance_rate,
                   M = M, who_eats_who = who_eats_who,
                   forager_or_not = forager_or_not, layered_foraging = layered_foraging, efficiency = efficiency)

    return payoff_i


#@njit
def lin_growth_jit(resources, populations, i, j, strategy, current_layered_attack = None,
                   heat_kernel = None, clearance_rate = None,
                   M = None, who_eats_who = None,
                   forager_or_not = None, layered_foraging = None, efficiency = None):

    x_temp = np.copy(strategy)
    x_temp[0] = x_temp[0] @ heat_kernel  # Going smooth.

    x_temp[1] = x_temp[1] @ heat_kernel  # Going smooth.

    predator_hunger = clearance_rate[j] * populations[j] * np.dot(M, current_layered_attack[:, j, i] * x_temp[1]) * who_eats_who[j, i]

    x = x_temp[0].reshape(-1, 1)
    interaction_term = who_eats_who[i, j] * populations[j] * clearance_rate[i]
    lin_growth = interaction_term * (x_temp[1] * current_layered_attack[:, i, j].T) @ M @ x

    foraging_term_self = (resources * forager_or_not[i] *
                          clearance_rate[i] * layered_foraging[:, i]).reshape(
        (1, -1)) @ (M @ x)
    foraging_term_self = foraging_term_self / (populations.size - 1)

    actual_growth = efficiency * (lin_growth + foraging_term_self)

    pred_loss = x.T @ predator_hunger

    return actual_growth - pred_loss

@njit(fastmath=True)
def jit_payoff_matrix_builder(resources, populations, i, j, current_layered_attack,
                   heat_kernel = None, clearance_rate = None,
                   M = None, who_eats_who = None,
                   forager_or_not = None, layered_foraging = None, efficiency = None):
    layers = current_layered_attack.shape[0]
    payoff_i = np.zeros((current_layered_attack.shape[0], current_layered_attack.shape[0]))
    for k in range(current_layered_attack.shape[0]):
        one_k_vec = np.zeros(layers)
        one_k_vec[k] = 1
        for n in range(layers):
            one_n_vec = np.zeros(layers)
            one_n_vec[n] = 1
            strat_mat = np.vstack((one_k_vec, one_n_vec))
            x_temp = np.copy(strat_mat)
            x_temp[0] = x_temp[0] @ heat_kernel  # Going smooth.

            x_temp[1] = x_temp[1] @ heat_kernel  # Going smooth.

            predator_hunger = clearance_rate[j] * populations[j] * np.dot(M, current_layered_attack[:, j, i] * x_temp[1]) * \
                              who_eats_who[j, i]

            x = x_temp[0].reshape(-1, 1)
            interaction_term = who_eats_who[i, j] * populations[j] * clearance_rate[i]
            lin_growth = interaction_term * (x_temp[1] * current_layered_attack[:, i, j].T) @ M @ x

            foraging_term_self = (resources * forager_or_not[i] *
                                  clearance_rate[i] * layered_foraging[:, i]).reshape(
                (1, -1)) @ (M @ x)
            foraging_term_self = foraging_term_self / (populations.size - 1)

            actual_growth = efficiency * (lin_growth[0] + foraging_term_self[0])

            pred_loss = x.T @ predator_hunger
            #print(pred_loss, actual_growth)

            payoff_i[k, n] = actual_growth[0] - pred_loss[0]

            #lin_growth_jit(resources, populations, i, j, strat_mat, current_layered_attack = current_layered_attack,
                   #heat_kernel = heat_kernel, clearance_rate = clearance_rate,
                   #M = M, who_eats_who = who_eats_who,
                   #forager_or_not = forager_or_not, layered_foraging = layered_foraging, efficiency = efficiency).astype(float)

    return payoff_i

def lin_growth_no_pops_no_res(eco, i, j, layered_attack, strategy):
    x_temp = np.copy(strategy)
    x_temp[0] = x_temp[0] @ eco.heat_kernels[i]  # Going smooth.

    x_temp[1] = x_temp[1] @ eco.heat_kernels[j]  # Going smooth.

    x = x_temp[0].reshape(-1, 1)
    interaction_term = eco.parameters.who_eats_who[i, j] * eco.parameters.clearance_rate[i]
    lin_growth = interaction_term * (x_temp[1] * layered_attack[:, i, j].T) @ eco.spectral.M @ x

    actual_growth = lin_growth

    return actual_growth
