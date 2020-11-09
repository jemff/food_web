from utility_functions import  *
import matplotlib.pyplot as plt

def lemke_optimizer_sparse(eco, payoff_matrix = None, dirac_mode = True):
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

def total_payoff_matrix_builder_sparse(eco, current_layered_attack = None, dirac_mode = False):
    total_payoff_matrix = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))

    if current_layered_attack is None:
        current_layered_attack = eco.parameters.layered_attack

    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            if i != j:
                i_vs_j = payoff_matrix_builder(eco, i, j, current_layered_attack = current_layered_attack, dirac_mode = dirac_mode)
            elif i == j:
                i_vs_j = np.zeros((eco.layers, eco.layers))
            #if i == 1:
            #    total_payoff_matrix[i*eco.layers:(i+1)*eco.layers, j*eco.layers: (j+1)*eco.layers] = i_vs_j.T
            #else:

            total_payoff_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = i_vs_j
#    print("MAXIMM PAYDAY ORIGINAL",  np.max(total_payoff_matrix))
    total_payoff_matrix[total_payoff_matrix != 0] = total_payoff_matrix[total_payoff_matrix != 0] - np.max(total_payoff_matrix) #- 1 #Making sure everything is negative  #- 0.00001
    #total_payoff_matrix = total_payoff_matrix/np.max(-total_payoff_matrix)
    print(np.where(total_payoff_matrix == 0))
    return total_payoff_matrix

depth = 45
layers = 100
segments = 1
size_classes = 2
lam = 100
simulate = False
verbose = True
l2 = False
min_attack_rate = 10**(-3)
mass_vector = np.array([0.05, 0.05*408])  # np.array([1, 30, 300, 400, 800, 16000])

obj = spectral_method(depth, layers, segments=segments)
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc=0, scale=3)
res_start = 4*norm_dist  # 0.1*(1-obj.x/depth)
res_max = 8*norm_dist

water_start = water_column(obj, res_start, layers=layers * segments, resource_max=res_max, replacement=lam, advection=0,
                           diffusion=0, logistic = True)

params = ecosystem_parameters(mass_vector, obj, lam=0.3, min_attack_rate = min_attack_rate, forage_mass = 0.05/408)
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers * segments, params, obj, water_start, l2=l2, movement_cost=0)
eco.population_setter(np.array([1, 0.05]))

eco.heat_kernel_creator(10**(-1))
eco.heat_kernels[1] = eco.heat_kernels[0]

S = lemke_optimizer_sparse(eco, total_payoff_matrix_builder_sparse(eco))
S1 = lemke_optimizer(eco)
plt.plot(eco.spectral.x, S[0:layers]@eco.heat_kernels[0])
plt.plot(eco.spectral.x, S[layers:2*layers]@eco.heat_kernels[0])

S2 = quadratic_optimizer(eco, payoff_matrix=total_payoff_matrix_builder_sparse(eco))

S3 = quadratic_optimizer(eco)

plt.show()

plt.plot(eco.spectral.x, S1[0:layers]@eco.heat_kernels[0])
plt.plot(eco.spectral.x, S1[layers:2*layers]@eco.heat_kernels[0])
plt.show()

plt.plot(eco.spectral.x, S2[0:layers]@eco.heat_kernels[0])
plt.plot(eco.spectral.x, S2[layers:2*layers]@eco.heat_kernels[0])
plt.show()

plt.plot(eco.spectral.x, S3[0:layers]@eco.heat_kernels[0])
plt.plot(eco.spectral.x, S3[layers:2*layers]@eco.heat_kernels[0])
plt.show()