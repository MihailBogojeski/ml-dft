from ml_dft.multivariate_gpr_cv import MultivariateGaussianProcessCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os.path
from sacred import Experiment, cli_option
from ml_dft import dft_utils as du
from sacred.observers import MongoObserver
import numpy as np
import time


@cli_option('-M', '--mongo_db')
def mongo_option(args, run):
    # run.config contains the configuration. You can read from there.
    split_args = args.split(';')
    print(split_args)
    mongo = MongoObserver.create(url=split_args[0], db_name=split_args[1],
                                 collection=None if split_args[2] == '' else split_args[2])
    run.observers.append(mongo)


ex = Experiment('Machine learning DFT', additional_cli_options=[mongo_option])


@ex.config
def config():
    data_dir = '.'
    train_dir = 'water_102'
    test_dir = 'water_102'
    # choices=['train', 'test', 'calculate_potentials',
    #          'potentials_to_density', 'density_to_energy',
    #          'predict_density', 'potentials_to_energy',
    #          'test_desc_en', 'make_plots', 'test_cc',
    #          'test_desc_en_cc', 'transform_positions'])
    train_inds = list(range(0, 50))
    test_inds = list(range(50, 102))
    cv_inds = None
    train_inds_file = None
    test_inds_file = None
    cv_inds_file = None
    run_id = 1
    output_file = None
    dist_file = None
    dist_save = None
    energy_type = 'cc'
    descriptor_type = 'pot'
    gaussian_width = 0.6051106340808006
    grid_spacing = 0.33109039923721575
    grid_file = None
    density_kernel = 'rbf'
    energy_kernel = 'rbf'
    density_kernel_params = {}
    energy_kernel_params = {}
    density_alpha_params = [2.21221629e-14]
    density_gamma_params = list(1 / np.sqrt(2 * np.logspace(-10, -1, 10)))
    density_lambda_params = list([0.1, 0.25, 0.5, 1])
    energy_alpha_params = [2.21221629e-14]
    energy_gamma_params = list(1 / np.sqrt(2 * np.logspace(-15, -5, 10)))
    energy_lambda_params = list([0.1, 0.25, 0.5, 1])
    use_true_densities = False
    verbose = 1
    importance_weighting = False

    train_prob_inds = 'train_prob_inds.npy'
    test_prob_inds = 'test_prob_inds.npy'
    eval_ratio_inds = 'eval_ratio_inds.npy'

    # matern kernel
    # density_alpha_params = list(np.logspace(-100, 0, 10))
    # density_gamma_params = list(1 / np.sqrt(2 * np.logspace(-20, 0, 10)))
    # energy_alpha_params = list(np.logspace(-120, 0, 30))
    # energy_gamma_params = list(1 / np.sqrt(2 * np.logspace(-30, 0, 30)))
    plot_cv_errors = False

    if verbose > 0:
        print('Parameters:')
        print('data_dir:', data_dir)
        print('train_dir:', train_dir)
        print('test_dir:', test_dir)
        print('run_id:', run_id)
        print('output_file', output_file)
        print('energy_type', energy_type)
        print('descriptor_type', descriptor_type)
        print('grid_spacing:', grid_spacing)
        print('gaussian_width:', gaussian_width)
        print('grid_file:', grid_file)
        print('train_inds_file:', train_inds_file)
        print('test_inds_file:', test_inds_file)
        print('cv_inds_file:', cv_inds_file)
        print('density_kernel:', density_kernel)
        print('energy_kernel:', energy_kernel)
        print('density_kernel_params:', density_kernel_params)
        print('energy_kernel_params:', energy_kernel_params)
        print('density_alpha_params:', density_alpha_params)
        print('density_gamma_params:', density_gamma_params)
        print('density_lambda_params:', density_lambda_params)
        print('energy_alpha_params:', energy_alpha_params)
        print('energy_gamma_params:', energy_gamma_params)
        print('energy_lambda_params:', energy_lambda_params)
        print('dist_file:', dist_file)
        print('dist_save:', dist_save)
        print('plot_cv_errors:', plot_cv_errors)
        print('use_true_densities:', use_true_densities)
        print('verbose:', verbose)
        print('importance_weighting', importance_weighting)
    if verbose > 1:
        print('train_inds:', train_inds)
        print('test_inds:', test_inds)
        print('cv_inds', cv_inds)
        print('train_prob_inds', train_prob_inds)
        print('test_prob_inds', test_prob_inds)
        print('eval_ratio_inds', eval_ratio_inds)


@ex.command
def transform_positions(data_dir, train_dir):
    """Performs a transformation operation on the positions of the atoms
    in order to align them.

    Args:
        config: Dictionary containing the necessary configuration parameters.
    """
    positions = np.load(os.path.join(data_dir, train_dir, 'structures.npy'))
    charges = np.load(os.path.join(data_dir, train_dir, 'atom_types.npy'))
    base = np.load(os.path.join(data_dir, train_dir, 'base_pos.npy'))
    heavy = []
    if len(charges) <= 3:
        heavy = np.ones(len(charges), dtype=np.bool_)
    else:
        heavy = np.where(charges > 1)[0]
        heavy = np.array(heavy, dtype=np.bool_)

    for i in range(positions.shape[0]):
        positions[i] = du.transform_molecule(positions[i], base, heavy)

    pos_new = positions

    np.save(os.path.join(data_dir, train_dir, 'trans_pos.npy'), pos_new)

    return pos_new


@ex.command
def estimate_descriptor_importance_weights(data_dir, train_dir, run_id, train_prob_inds, test_prob_inds,
                                           eval_ratio_inds, descriptor_type, verbose,
                                           density_gamma_params):
    train_prob_inds = np.load(train_prob_inds)
    test_prob_inds = np.load(test_prob_inds)
    eval_ratio_inds = np.load(eval_ratio_inds)

    if verbose > 0:
        print('Loading descriptors')
    descriptors = np.load(os.path.join(data_dir, train_dir, descriptor_type + '_' + str(run_id) + '.npy'))
    if verbose > 0:
        print('Loaded descriptors')

    train_dens = descriptors[train_prob_inds]
    test_dens = descriptors[test_prob_inds]
    eval_dens = descriptors[eval_ratio_inds]
    # ratios = du.estimate_te_tr_likelihood_ratio(train_dens, test_dens, eval_dens,
    #                                             train_norm_factor,
    #                                             test_norm_factor,
    #                                             eval_norm_factor)
    if verbose > 0:
        print('Estimating importance weights')
    ratios = du.estimate_importance_weights(train_dens, test_dens, eval_dens, list(density_gamma_params))
    if verbose > 0:
        print('importance weights:', ratios)
        print('saving to:', os.path.join(data_dir, train_dir, 'importance_weights_' + descriptor_type + '_' + str(run_id) + '.npy'))

    np.save(os.path.join(data_dir, train_dir, 'importance_weights_' + descriptor_type + '_' + str(run_id) + '.npy'), ratios)


@ex.command
def estimate_density_importance_weights(data_dir, train_dir, run_id, train_prob_inds, test_prob_inds,
                                        eval_ratio_inds, use_true_densities, verbose,
                                        energy_gamma_params):
    train_prob_inds = np.load(train_prob_inds)
    test_prob_inds = np.load(test_prob_inds)
    eval_ratio_inds = np.load(eval_ratio_inds)

    if use_true_densities:
        densities = np.load(os.path.join(data_dir, train_dir, 'densities.npy'))
    else:
        densities = np.load(os.path.join(data_dir, train_dir, 'coefs_pred_' + str(run_id) + '.npy'))

    train_dens = densities[train_prob_inds]
    test_dens = densities[test_prob_inds]
    eval_dens = densities[eval_ratio_inds]
    # ratios = du.estimate_te_tr_likelihood_ratio(train_dens, test_dens, eval_dens,
    #                                             train_norm_factor,
    #                                             test_norm_factor,
    #                                             eval_norm_factor)
    ratios = du.estimate_importance_weights(train_dens, test_dens, eval_dens, list(energy_gamma_params))
    if use_true_densities:
        suffix = 'true'
    else:
        suffix = 'pred'
    if verbose > 0:
        print('importance weights:', ratios)
        print('saving to:', os.path.join(data_dir, train_dir, 'importance_weights_dens_' + suffix + '_' + str(run_id) + '.npy'))

    np.save(os.path.join(data_dir, train_dir, 'importance_weights_dens_' + suffix + '_' + str(run_id) + '.npy'), ratios)


@ex.command
def calculate_descriptors(descriptor_type):
    if descriptor_type == 'pot':
        calculate_potentials()


@ex.command
def calculate_potentials(data_dir, train_dir, test_dir, run_id, grid_spacing, grid_file, gaussian_width, verbose):
    """Calculates the potential as a sum of artificial Gaussians
       based on the atom positions as saves them on disk.

    Args:
        config: Dictionary containing the necessary configuration parameters.
    """

    if verbose >= 0:
        print('Calculating artificial potentials')
    if train_dir == test_dir:
        work_dirs = [train_dir]
    else:
        work_dirs = [train_dir, test_dir]

    all_positions = []
    for work_dir in work_dirs:
        positions = np.load(os.path.join(data_dir, work_dir, 'structures.npy'))
        all_positions.append(np.copy(positions))

        # positions = positions[config['id_range'], :]

    cat_positions = np.concatenate(all_positions, axis=0)
    if verbose > 0:
        print('Positions shape:', cat_positions.shape)

    max_pos = np.max(cat_positions, axis=0)
    max_pos = np.max(max_pos, axis=0)
    if verbose > 1:
        print(max_pos)

    min_pos = np.min(cat_positions, axis=0)
    min_pos = np.min(min_pos, axis=0)
    if verbose > 1:
        print(min_pos)

    max_pos = np.ceil(max_pos) + 1
    min_pos = np.floor(min_pos) - 1
    if verbose > 0:
        print('Min, max coordinates:', min_pos, max_pos)

    # max_pos = np.array([14.5, 15, 11])
    # min_pos = np.array([5, 4.5, 9])
    max_range = max_pos - min_pos

    steps = np.round(max_range / grid_spacing).astype(np.int)

    if (grid_file is None):
        grid_range = [np.linspace(ma, mm, s) for ma, mm, s in zip(max_pos, min_pos, steps)]
    elif(grid_file == ''):
        if verbose > 0:
            print('Saving potentials grid')
        grid_range = [np.linspace(ma, mm, s) for ma, mm, s in zip(max_pos, min_pos, steps)]
        grid_range = np.array(grid_range, dtype=object)
        np.save(os.path.join(data_dir, train_dir, 'grid_range_' + str(run_id) + '.npy'), grid_range)
    elif os.path.exists(os.path.join(data_dir, train_dir, grid_file + '_' + str(run_id) + '.npy')):
        grid_range = np.load(os.path.join(data_dir, train_dir, grid_file + '_' + str(run_id) + '.npy'), allow_pickle=True)
    else:
        raise RuntimeError('Invalid grid file: \'' + grid_file + '\'')

    # grid_range = np.linspace(min_pos, max_pos, steps)

    Y, X, Z = np.meshgrid(grid_range[0], grid_range[1], grid_range[2])
    if verbose > 1:
        print(X.shape)
    X = X.flatten()[:, np.newaxis]
    Y = Y.flatten()[:, np.newaxis]
    Z = Z.flatten()[:, np.newaxis]

    grid = np.concatenate((X, Y, Z), axis=1)
    if verbose > 0:
        print('Potential positions shape', grid.shape)
    for i, work_dir in enumerate(work_dirs):
        charges = np.load(os.path.join(data_dir, work_dir, 'atom_types.npy'))
        positions = all_positions[i]

        potentials = du.calculate_potential(positions, charges, gaussian_width, grid, verbose=verbose)

        if verbose > 1:
            print(potentials.shape)
        np.save(os.path.join(data_dir, work_dir, 'pot_' + str(run_id) + '.npy'), potentials)


@ex.command
def descriptors_to_density(data_dir, train_dir, train_inds, train_inds_file, run_id, descriptor_type, density_alpha_params,
                           density_gamma_params, density_lambda_params, density_kernel, density_kernel_params,
                           plot_cv_errors, importance_weighting, cv_inds, cv_inds_file, verbose,
                           dist_file):
    """Trains independent KRR models to predict the basis coefficients of the density from
        the descriptors
    Args:
        config: Dictionary containing the necessary configuration parameters.
    """
    if verbose > 0:
        print('Training descriptors to density model')
    if train_inds_file is not None:
        train_inds = list(np.load(train_inds_file))
        # print('train inds', train_inds)
    if cv_inds_file is not None:
        cv_inds = np.load(cv_inds_file)
    descriptors = np.load(os.path.join(data_dir, train_dir, descriptor_type + '_' + str(run_id) + '.npy'))
    coefs = np.load(os.path.join(data_dir, train_dir, 'dft_densities.npy'))
    if importance_weighting:
        importance_weights = np.load(os.path.join(data_dir, train_dir, 'importance_weights_' +
                                                  descriptor_type + '_' + str(run_id) + '.npy'))
        print(importance_weights)
        importance_weights = importance_weights[train_inds]
    else:
        importance_weights = None

    descriptors = descriptors[train_inds, :]
    coefs = coefs[train_inds, :]
    if verbose > 1:
        print(coefs.shape)
    dist_mat = None
    if dist_file:
        dist_file = os.path.join(data_dir, train_dir, dist_file + '.npy')
        if os.path.exists(dist_file):
            if verbose > 0:
                print('Loading distance matrix from file:', dist_file)
            dist_mat = np.load(dist_file)

    density_kr = MultivariateGaussianProcessCV(cv_nfolds=5,
                                               krr_param_grid={"alpha": density_alpha_params,
                                                               "gamma": density_gamma_params,
                                                               "lambda": density_lambda_params},
                                               id=run_id,
                                               verbose=verbose,
                                               kernel=density_kernel,
                                               kernel_params=density_kernel_params)

    start = time.time()
    density_kr.fit(descriptors, coefs, importance_weights=importance_weights, cv_indices=cv_inds,
                   dist=dist_mat, dist_savename=dist_file)
    end = time.time()
    if verbose > 1:
        print('Elapsed train', end - start)
    if plot_cv_errors:
        density_kr.plot_cv_error()
    density_kr.save(os.path.join(data_dir, train_dir, 'density_kr_' + str(run_id)))

    if verbose > 1:
        print('Alphas = ' + str(density_kr.alphas_))
        print('Gammas = ' + str(density_kr.gammas_))
    ex.info['density_kr_alpha'] = density_kr.alphas_[0]
    ex.info['density_kr_gamma'] = density_kr.gammas_[0]
    start = time.time()
    # np.save(os.path.join(data_dir, train_dir, 'density_kr_' + str(run_id) + '.npy'), density_kr)
    end = time.time()
    if verbose > 1:
        print('Elapsed save', end - start)

    return np.mean(density_kr.errors)


@ex.command
def predict_density(data_dir, train_dir, test_dir, run_id, descriptor_type,
                    density_alpha_params, density_gamma_params, verbose, dist_file):
    """Predicts the density coefficients using the learned KRR models
    Args:
        config: Dictionary containing the necessary configuration parameters.
    """
    if verbose > 0:
        print('Predicting density coefficients')
    dist_mat = None
    if dist_file:
        dist_file = os.path.join(data_dir, train_dir, dist_file + '.npy')
        if os.path.exists(dist_file):
            if verbose > 0:
                print('Loading distance matrix from file:', dist_file)
            dist_mat = np.load(dist_file)
    for work_dir in [train_dir, test_dir]:
        descriptors = np.load(os.path.join(data_dir, work_dir, descriptor_type + '_' + str(run_id) + '.npy'))
        coefs = np.load(os.path.join(data_dir, work_dir, 'dft_densities.npy'))

        if verbose > 1:
            print(coefs.shape)

        density_kr = MultivariateGaussianProcessCV(cv_nfolds=5,
                                                   krr_param_grid={"alpha": density_alpha_params,
                                                                   "gamma": density_gamma_params},
                                                   id=run_id,
                                                   verbose=verbose,
                                                   )
        density_kr.load(os.path.join(data_dir, train_dir, 'density_kr_' + str(run_id)))

        start = time.time()
        if work_dir == train_dir:
            dist = dist_mat
        else:
            dist = None
        coefs_pred = density_kr.predict(descriptors, dist=dist)
        end = time.time()
        if verbose > 1:
            print('Elapsed predict', end - start)

        start = time.time()
        end = time.time()
        if verbose > 1:
            print('Elapsed save', end - start)

        if verbose > 0:
            print('RMSE:', np.mean(np.linalg.norm(coefs[:len(coefs_pred)] - coefs_pred, axis=1)))
            print('Coefs norm:', np.mean(np.linalg.norm(coefs, axis=1)))
            print('Coef predictions norm:', np.mean(np.linalg.norm(coefs_pred, axis=1)))

        np.save(os.path.join(data_dir, work_dir, 'coefs_pred_' + str(run_id) + '.npy'), coefs_pred)

    return np.mean(np.linalg.norm(coefs[:len(coefs_pred)] - coefs_pred, axis=1))


@ex.command
def density_to_energy(data_dir, train_dir, energy_type, train_inds, train_inds_file, run_id,
                      energy_alpha_params, energy_gamma_params, energy_lambda_params, energy_kernel, energy_kernel_params,
                      plot_cv_errors, use_true_densities, importance_weighting, cv_inds, cv_inds_file, verbose,
                      dist_file):
    """Trains a KRR model that predicts the energy from the density coefficients
    Args:
        config: Dictionary containing the necessary configuration parameters.
    """
    if verbose > 0:
        print('Training density to energy model')

    if use_true_densities:
        coefs_pred = np.load(os.path.join(data_dir, train_dir, 'dft_densities.npy'))
    else:
        coefs_pred = np.load(os.path.join(data_dir, train_dir, 'coefs_pred_' + str(run_id) + '.npy'))

    if train_inds_file is not None:
        train_inds = list(np.load(train_inds_file))
    if cv_inds_file is not None:
        cv_inds = np.load(cv_inds_file)
    if use_true_densities:
        suffix = 'true'
    else:
        suffix = 'pred'
    if importance_weighting:
        importance_weights = np.load(os.path.join(data_dir, train_dir, 'importance_weights_dens_' + suffix + '_' + str(run_id) + '.npy'))
        print(importance_weights.shape)
        importance_weights = importance_weights[train_inds]
    else:
        importance_weights = None

    energies = np.load(os.path.join(data_dir, train_dir, energy_type + '_energies.npy'))
    dist_mat = None
    if dist_file:
        dist_file = os.path.join(data_dir, train_dir, dist_file + '.npy')
        if os.path.exists(dist_file):
            if verbose > 0:
                print('Loading distance matrix from file:', dist_file)
            dist_mat = np.load(dist_file)

    coefs_pred = coefs_pred[train_inds, :]
    energies = energies[train_inds]
    energies = np.reshape(energies, (-1, 1))
    if verbose > 1:
        print(energies.shape)

    energy_kr = MultivariateGaussianProcessCV(cv_nfolds=5,
                                              krr_param_grid={"alpha": energy_alpha_params,
                                                              "gamma": energy_gamma_params,
                                                              "lambda": energy_lambda_params},
                                              id=run_id + .5,
                                              verbose=verbose,
                                              kernel=energy_kernel,
                                              kernel_params=energy_kernel_params,
                                              delta_learning=(energy_type == 'diff'),
                                              mae=True)
    energy_kr.fit(coefs_pred, energies, importance_weights=importance_weights, cv_indices=cv_inds,
                  dist=dist_mat, dist_savename=dist_file)

    energy_kr.save(os.path.join(data_dir, train_dir,
                                str(energy_type) + str('_kr_') + str(run_id)))

    if plot_cv_errors:
        energy_kr.plot_cv_error()

    if dist_file and dist_mat is None:
        if os.path.exists(dist_file):
            if verbose > 0:
                print('Loading distance matrix from file:', dist_file)
            dist_mat = np.load(dist_file)

    energy_pred = energy_kr.predict(coefs_pred, dist=dist_mat)
    start = time.time()
    end = time.time()
    if verbose > 1:
        print('Elapsed save', end - start)
    if verbose > 1:
        print('Alphas = ' + str(energy_kr.alphas_))
        print('Gammas = ' + str(energy_kr.gammas_))
    ex.info['energy_kr_alpha'] = energy_kr.alphas_[0]
    ex.info['energy_kr_gamma'] = energy_kr.gammas_[0]
    # print(energies)
    # print(energy_pred)
    if verbose > 0:
        print('Energy training error:', mean_absolute_error(energies, energy_pred))
    return(mean_absolute_error(energies, energy_pred))


@ex.command
def descriptors_to_energy(data_dir, train_dir, energy_type, train_inds, train_inds_file, run_id,
                          energy_alpha_params, energy_gamma_params, energy_lambda_params, energy_kernel, descriptor_type,
                          energy_kernel_params, plot_cv_errors, importance_weighting,
                          cv_inds, cv_inds_file, verbose):
    """Trains a KRR model that predicts the energy from the density coefficients
    Args:
        config: Dictionary containing the necessary configuration parameters.
    """
    if verbose > 0:
        print('Training descriptors to energy model')

    if train_inds_file is not None:
        train_inds = list(np.load(train_inds_file))

    if cv_inds_file is not None:
        cv_inds = np.load(cv_inds_file)

    descriptors = np.load(os.path.join(data_dir, train_dir, descriptor_type + '_' + str(run_id) + '.npy'))
    energies = np.load(os.path.join(data_dir, train_dir, energy_type + '_energies.npy'))
    if importance_weighting:
        importance_weights = np.load(os.path.join(data_dir, train_dir, 'importance_weights_' +
                                                  descriptor_type + '_' + str(run_id) + '.npy'))
        print(importance_weights)
        importance_weights = importance_weights[train_inds]
    else:
        importance_weights = None

    descriptors = descriptors[train_inds, :]
    energies = energies[train_inds]
    energies = np.reshape(energies, (-1, 1))
    if verbose > 1:
        print(energies.shape)
    if verbose > 1:
        print(descriptors.shape)

    energy_pot_kr = MultivariateGaussianProcessCV(cv_nfolds=5,  # shuffles=10,
                                                  krr_param_grid={"alpha": energy_alpha_params,
                                                                  "gamma": energy_gamma_params,
                                                                  "lambda": energy_lambda_params},
                                                  id=run_id + .5,
                                                  verbose=verbose,
                                                  kernel=energy_kernel,
                                                  kernel_params=energy_kernel_params,
                                                  delta_learning=(energy_type == 'diff'))
    energy_pot_kr.fit(descriptors, energies, importance_weights=importance_weights, cv_indices=cv_inds)

    if plot_cv_errors:
        energy_pot_kr.plot_cv_error()

    start = time.time()
    energy_pot_kr.save(os.path.join(data_dir, train_dir,
                                    energy_type + '_pot_kr_' + str(run_id)))
    end = time.time()
    if verbose > 1:
        print('Elapsed save', end - start)
    energy_pred = energy_pot_kr.predict(descriptors)
    # energy_pred = np.reshape(energy_pred, energies.shape)
    # print(energies)
    # print(energy_pred)
    if verbose > 0:
        print('Energy train error:', mean_absolute_error(energies, energy_pred))
    # print('Energy test error:', np.mean(np.abs(energies - energy_pred)))
    if verbose > 1:
        print('Alphas = ' + str(energy_pot_kr.alphas_))
        print('Gammas = ' + str(energy_pot_kr.gammas_))

    return mean_absolute_error(energies, energy_pred)


@ex.command
def test(data_dir, train_dir, test_dir, energy_type, test_inds, test_inds_file, run_id, output_file,
         energy_alpha_params, energy_gamma_params, use_true_densities, _config, verbose):
    """Evaluates the model using the data in id_range
        config: Dictionary containing the necessary configuration parameters.
    """
    print('Evaluating the model')
    if verbose > 0:
        print('Testing the model')
    if test_inds_file is not None:
        test_inds = list(np.load(test_inds_file))
    energies = np.load(os.path.join(data_dir, test_dir, energy_type + '_energies.npy'))

    if verbose > 0:
        print('Loading densities')
    if use_true_densities:
        coefs_pred = np.load(os.path.join(data_dir, test_dir, 'dft_densities.npy'))
    else:
        coefs_pred = np.load(os.path.join(data_dir, test_dir, 'coefs_pred_' + str(run_id) + '.npy'))

    coefs_pred = coefs_pred[test_inds, :]
    energies = energies[test_inds]
    energies = np.reshape(energies, (-1, 1))

    if verbose > 0:
        print('Loading energy kernel')
    energy_kr = MultivariateGaussianProcessCV(cv_nfolds=5,
                                              krr_param_grid={"alpha": energy_alpha_params,
                                                              "gamma": energy_gamma_params},
                                              id=run_id + .5,
                                              verbose=verbose,
                                              )
    energy_kr.load(os.path.join(data_dir, train_dir, energy_type + '_kr_' + str(run_id)))

    if verbose > 0:
        print('Predicting energies')
    energies_pred = energy_kr.predict(coefs_pred)

    energies_pred = np.reshape(energies_pred, (-1, 1))

    errors_pred = np.abs(energies - energies_pred)
    ex.info['corr'] = np.corrcoef(energies.T, energies_pred.T)[0][1]
    ex.info['errors'] = errors_pred
    ex.info['preds'] = energies_pred
    print('Energies:')
    print('Correlation: ', np.corrcoef(energies.T, energies_pred.T)[0][1])
    print('RMSE: ', np.sqrt(mean_squared_error(energies, energies_pred)))
    print('MAE: ', mean_absolute_error(energies, energies_pred))
    print('Max MAE: ', np.max(np.abs(energies - energies_pred)))
    if output_file is not None:
        if verbose > 0:
            print('Writing to file:', os.path.join(data_dir, test_dir,
                                                   output_file + '_' + energy_type +
                                                   '_' + str(run_id) + '.npy'))
        f = open(os.path.join(data_dir, test_dir,
                              output_file + '_' + energy_type + '_' + str(run_id) + '.npy'), 'w')
        f.write('Config:\n')
        f.write(str(_config))
        f.write('\n')
        f.write('Training sample inds:' + str(test_inds) + '\n')
        f.write('Energies\n')
        corr = np.corrcoef(energies.T, energies_pred.T)[0][1]
        if np.isnan(corr):
            corr = 0
        if verbose > 0:
            print('Correlation is ', corr)
        f.write('Correlation: ' + str(corr) + '\n')
        f.write('RMSE: ' + str(np.sqrt(mean_squared_error(energies, energies_pred))) + '\n')
        f.write('MAE: ' + str(mean_absolute_error(energies, energies_pred)) + '\n')
        f.write('Max MAE: ' + str(np.max(np.abs(energies - energies_pred))) + '\n')
        np.save(os.path.join(data_dir, test_dir,
                             'errors_pred_' + energy_type +
                             '_' + str(run_id) + '.npy'), errors_pred)

    return mean_absolute_error(energies, energies_pred)


@ex.command
def test_desc_en(data_dir, train_dir, test_dir, energy_type, test_inds, test_inds_file, descriptor_type,
                 run_id, output_file, _config, verbose):
    """Evaluates the model using the data in id_range
        config: Dictionary containing the necessary configuration parameters.
    """
    if verbose >= 0:
        print('Testing the model')
    descriptors = np.load(os.path.join(data_dir, test_dir, descriptor_type + '_' + str(run_id) + '.npy'))
    energies = np.load(os.path.join(data_dir, test_dir, energy_type + '_energies.npy'))

    if test_inds_file is not None:
        test_inds = list(np.load(test_inds_file))

    energies = energies[test_inds]
    descriptors = descriptors[test_inds, :]
    energies = np.reshape(energies, (-1, 1))

    energy_pot_kr = MultivariateGaussianProcessCV(cv_nfolds=5,  # shuffles=10,
                                                  krr_param_grid={"alpha": np.logspace(-20, 0, 10),
                                                                  "gamma": np.logspace(-10, 0, 10)},
                                                  id=run_id + .5,
                                                  verbose=verbose,
                                                  )
    energy_pot_kr.load(os.path.join(data_dir, train_dir,
                                    energy_type + '_pot_kr_' + str(run_id)))

    # energies_pred = energy_pot_kr.predict(coefs)
    energies_pred = energy_pot_kr.predict(descriptors)

    energies_pred = np.reshape(energies_pred, (-1, 1))

    errors_pred = np.abs(energies - energies_pred)
    ex.info['errors'] = errors_pred
    ex.info['corr'] = np.corrcoef(energies.T, energies_pred.T)[0][1]
    if output_file is None:
        print('Energies:')
        print('Correlation: ', np.corrcoef(energies.T, energies_pred.T)[0][1])
        print('RMSE: ', np.sqrt(mean_squared_error(energies, energies_pred)))
        print('MAE: ', mean_absolute_error(energies, energies_pred))
        print('Max: ', np.max(np.abs(energies - energies_pred)))
    else:
        if verbose > 0:
            print('Writing to file:', os.path.join(data_dir, test_dir,
                                                   output_file + '_desc_en_' +
                                                   energy_type + '_' + str(run_id) + '.npy'))
        f = open(os.path.join(data_dir, test_dir,
                              output_file + '_desc_en_' + energy_type + '_' + str(run_id) + '.npy'), 'w')
        f.write('Config:\n')
        f.write(str(_config))
        f.write('\n')
        f.write('Training sample inds:' + str(test_inds) + '\n')
        f.write('Energies\n')
        corr = np.corrcoef(energies.T, energies_pred.T)[0][1]
        if np.isnan(corr):
            corr = 0
        if verbose > 0:
            print('Correlation is ', corr)
        f.write('Correlation: ' + str(corr) + '\n')
        f.write('RMSE: ' + str(np.sqrt(mean_squared_error(energies, energies_pred))) + '\n')
        f.write('MAE: ' + str(mean_absolute_error(energies, energies_pred)) + '\n')
        f.write('Max: ' + str(np.max(np.abs(energies - energies_pred))) + '\n')

        np.save(os.path.join(data_dir, test_dir,
                             'errors_pred_desc_en_' + energy_type +
                             '_' + str(run_id) + '.npy'), errors_pred)

    return mean_absolute_error(energies, energies_pred)


@ex.command
def train():
    """Performs the entire training procedure to predict the energies
       config: Dictionary containing the necessary configuration parameters.
    """
    print('Starting training process')
    dens_cv_err = descriptors_to_density()
    ex.info['density_cv_err'] = dens_cv_err
    dens_err = predict_density()
    ex.info['density_test_err'] = dens_err
    en_err = density_to_energy()
    ex.info['energy_train_err'] = en_err


@ex.command
def train_full():
    """Performs the entire training procedure to predict the energies
        config: Dictionary containing the necessary configuration parameters.
    """
    calculate_descriptors()
    train()


@ex.command
def run_full():
    calculate_potentials()
    train()
    mae = test()

    return mae


@ex.command
def run_density():
    calculate_descriptors()
    err = descriptors_to_density()
    test_err = predict_density()

    ex.info['test_err'] = test_err

    return err


@ex.command
def run_energy():
    density_to_energy()
    mae = test()

    return mae


@ex.automain
def run_train_test():
    train()
    mae = test()

    return mae
