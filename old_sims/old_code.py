def constraint_builder(M, classes):
    lower_bound = np.zeros(M.shape[0]*classes)
    upper_bound = np.array([np.inf]*M.shape[0]*classes)
    identity_part = np.identity(M.shape[0]*classes)

    matrix_vec = M.sum(axis = 0)
#    print(matrix_vec)
    matrix = np.zeros((classes, M.shape[0]*classes))
    for i in range(classes):
        matrix[i, i*M.shape[0]: (i+1)*M.shape[0]] = matrix_vec

    one_bound = np.repeat(1, classes)

    bounds = optm.Bounds(lower_bound, upper_bound)

    return matrix, one_bound, bounds

def loss_func_coefficients(vec, size_classes = None, layers = None, spec = None, eco = None): #Remove, deprecated


    coeffs = vec.reshape((layers, size_classes))
    point_vec = np.dot(spec.vandermonde, coeffs)
    point_loss = eco.loss_function(point_vec.flatten())
    point_loss = point_loss.reshape((layers, size_classes)) #Attempt at finding minimum via. coefficients


    return np.sum(np.dot(point_loss.T, np.dot(spec.M, point_loss))) #np.sum((np.linalg.norm(loss_vec, axis = 0))**2) #np.sum(np.dot(loss_vec, np.dot(spec.M, loss_vec.T))) #

def loss_func(vec, size_classes = None, layers = None, spec = None): #Remove, deprecated

    loss_vec = np.reshape(vec, (size_classes, layers))

    return (np.sum((np.linalg.norm(loss_vec, axis=0))))**2  # np.sum(np.dot(loss_vec, np.dot(spec.M, loss_vec.T)))

def replacer_function(x): #Deprecated
    return np.array([x, 1-x]).squeeze()

def nash_refuge(eco, verbose = False): #Deprecated
    x_temp = np.copy(eco.strategy_matrix.flatten())  # np.zeros(size_classes*layers)
    x_temp2 = np.copy(eco.strategy_matrix.flatten())
    error = 1
    bounds1 = [(0.00000001, 1)] #optm.Bounds(np.array([0] * eco.layers), np.array([np.inf] * eco.layers))
    iterations = 0
    while error > 10 ** (-8) and iterations < 2:
        for k in range(eco.mass_vector.shape[0]):
            x_temp2[k * eco.layers] = optm.minimize(
                lambda x: -eco.one_actor_growth(eco.strategy_replacer(replacer_function(x), k, x_temp), k),
                x0=x_temp[eco.layers * k], bounds=bounds1).x
            x_temp2[k * eco.layers + 1] = 1 - x_temp2[k * eco.layers]

        if verbose is True:
            print("Error: ", np.max(np.abs(x_temp - x_temp2)))
        error = np.max(np.abs(x_temp - x_temp2))
        iterations += 1
        if iterations >= 100:
            print(iterations, "aaaw man, it failed :(", error, x_temp - x_temp2, eco.populations)

        x_temp = np.copy(x_temp2+x_temp)*1/2
    return x_temp