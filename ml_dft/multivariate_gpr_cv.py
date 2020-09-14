import numpy as np
import sys
import time
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator
from scipy.linalg import cholesky, solve_triangular
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from ml_dft.kernel_functions import RBFKernel, MaternKernel
import os
import warnings


def get_alpha_add(n_basis, n_grid, delta, v):
    alpha_add = np.pi * ((np.arange(n_basis / 2) / (n_grid * delta))**2 + v**2) / v
    alpha_add = np.repeat(alpha_add, 2)
    return alpha_add


class MultivariateGaussianProcessCV(BaseEstimator):
    def __init__(self, krr_param_grid=None, cv_type=None, cv_nfolds=5, cv_groups=None,
                 cv_shuffles=1, n_components=None, single_combo=True,
                 verbose=0, copy_X=True, v=None, n_basis=None, n_grid=None, delta=None,
                 id=1, cleanup=True, kernel=None, squared_dist=False, kernel_params=None,
                 delta_learning=False, mae=False, replace_fit=True):
        self.krr_param_grid = krr_param_grid
        self.verbose = verbose
        self.cv_nfolds = cv_nfolds
        self.cv_type = cv_type
        self.cv_groups = cv_groups
        self.cv_shuffles = cv_shuffles
        self.n_components = n_components
        self.single_combo = single_combo
        self.copy_X = copy_X
        self.n_grid = n_grid
        self.delta = delta
        self.n_basis = n_basis
        self.id = id
        self.cleanup = cleanup
        self.kernel = kernel
        self.squared_dist = squared_dist
        self.device = None
        self.replace_fit = replace_fit
        self.delta_learning = delta_learning
        self.mae = mae
        if self.kernel is None:
            self.kernel = RBFKernel()
        elif self.kernel == 'rbf':
            self.kernel = RBFKernel(**kernel_params)
        elif self.kernel == 'matern':
            self.kernel = MaternKernel(**kernel_params)

        if 'v' in self.krr_param_grid is not None and not single_combo:
            raise ValueError('Can only add to alpha if single_combo=True')

    def score(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y, labels=None, dist=None, importance_weights=None, cv_indices=None,
            dist_savename=None):
        t = time.time()

        if y.ndim < 2:
            y = y.reshape(-1, 1)

        if self.n_components is not None:
            if self.verbose > 0:
                elapsed = time.time() - t
                print('PCA [%dmin %dsec]' % (int(elapsed / 60),
                                             int(elapsed % 60)))
            sys.stdout.flush()
            self.pca = PCA(n_components=self.n_components, svd_solver='arpack')
            y_ = self.pca.fit_transform(y)
            if self.verbose > 0:
                print('Lost %.1f%% information ' % (self.pca.noise_variance_) +
                      '[%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))
                elapsed = time.time() - t
        else:
            y_ = y

        if labels is not None:
            raise RuntimeError('Not implemented.')

        if cv_indices is None:
            cv_indices = np.arange(X.shape[0])
        if self.cv_type is None:
            kfold = RepeatedKFold(n_splits=self.cv_nfolds, n_repeats=self.cv_shuffles)
            cv_folds = kfold.split(X[cv_indices])
            n_cv_folds = kfold.get_n_splits()
        elif self.cv_type == 'iter':
            cv_folds = self.cv_groups
            n_cv_folds = len(self.cv_groups)
        elif self.cv_type == 'group':
            groups = self.cv_groups
            if self.cv_nfolds is None:
                self.cv_nfolds = len(np.unique(groups))
            kfold = GroupKFold(n_splits=self.cv_nfolds)
            cv_folds = kfold.split(X[cv_indices], y[cv_indices], groups)
            n_cv_folds = kfold.get_n_splits()
        else:
            raise Exception('Cross-validation type not supported')

        add_train_inds = np.setdiff1d(np.arange(X.shape[0]), cv_indices)
        cv_folds = list(cv_folds)
        cv_folds = [(np.concatenate((train_fold, add_train_inds)), test_fold) for train_fold, test_fold in cv_folds]

        if self.verbose > 0:
            elapsed = time.time() - t
            print('Computing distance matrix [%dmin %dsec]' % (
                int(elapsed / 60), int(elapsed % 60)))
            sys.stdout.flush()

        if dist is None:
            dist = euclidean_distances(X, None, squared=self.squared_dist)
            if dist_savename is not None:
                if self.verbose > 0:
                    print('Saving distance matrix to file:', dist_savename)
                np.save(dist_savename, dist)

        if importance_weights is None:
            self.krr_param_grid['lambda'] = [0]
            importance_weights = np.ones((X.shape[0], ))

        importance_weights = importance_weights**(0.5)

        errors = []
        if 'v' in self.krr_param_grid:
            for fold_i, (train_i, test_i) in enumerate(cv_folds):
                fold_errors = np.empty((len(self.krr_param_grid['v']),
                                        len(self.krr_param_grid['gamma']),
                                        1,
                                        len(self.krr_param_grid['alpha']), y_.shape[1]))
                if self.verbose > 0:
                    elapsed = time.time() - t
                    print('CV %d of %d [%dmin %dsec]' % (fold_i + 1,
                                                         n_cv_folds,
                                                         int(elapsed / 60),
                                                         int(elapsed % 60)))
                    sys.stdout.flush()
                for v_i, v in enumerate(self.krr_param_grid['v']):
                    for gamma_i, gamma in enumerate(self.krr_param_grid['gamma']):
                        for lamb_i, lamb in enumerate(self.krr_param_grid['lambda']):
                            iw = importance_weights**lamb
                            iw = iw[:, None]
                            K_train = self.kernel.apply_to_dist(dist[np.ix_(train_i, train_i)], gamma=gamma)
                            K_train *= np.outer(iw[train_i], iw[train_i])
                            K_test = self.kernel.apply_to_dist(dist[np.ix_(test_i, train_i)], gamma=gamma)
                        if self.verbose > 0:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                            for alpha_i, alpha in enumerate(self.krr_param_grid['alpha']):
                                if self.verbose > 0:
                                    sys.stdout.write(',')
                                    sys.stdout.flush()
                                for y_i in np.arange(y_.shape[1]):
                                    K_train_ = K_train.copy()
                                    alpha_add = get_alpha_add(self.n_basis, self.n_grid, self.delta, v)
                                    K_train_.flat[::K_train_.shape[0] + 1] += alpha * alpha_add[y_i]
                                    try:
                                        L_ = cholesky(K_train_, lower=True)
                                        x = solve_triangular(L_, y_[train_i, y_i], lower=True)
                                        dual_coef_ = solve_triangular(L_.T, x)
                                        pred_mean = np.dot(K_test, dual_coef_)
                                        if self.mae:
                                            e = np.mean(np.abs(pred_mean - y_[test_i, y_i]), 0)
                                        else:
                                            e = np.mean((pred_mean - y_[test_i, y_i]) ** 2, 0)
                                    except np.linalg.LinAlgError:
                                        e = np.inf
                                    fold_errors[v_i, gamma_i, 0, alpha_i, y_i] = e
                if self.verbose > 0:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                errors.append(fold_errors)
            errors = np.array(errors)
            errors = np.mean(errors, 0)  # average over folds
        else:
            for fold_i, (train_i, test_i) in enumerate(cv_folds):
                fold_errors = np.empty((len(self.krr_param_grid['gamma']),
                                        len(self.krr_param_grid['lambda']),
                                        len(self.krr_param_grid['alpha']), y_.shape[1]))
                if self.verbose > 0:
                    elapsed = time.time() - t
                    print('CV %d of %d [%dmin %dsec]' % (fold_i + 1,
                                                         n_cv_folds,
                                                         int(elapsed / 60),
                                                         int(elapsed % 60)))
                    sys.stdout.flush()
                for gamma_i, gamma in enumerate(self.krr_param_grid['gamma']):
                    if self.verbose > 0:
                        sys.stdout.write('.')
                        sys.stdout.flush()
                    for lamb_i, lamb in enumerate(self.krr_param_grid['lambda']):
                        iw = importance_weights**lamb
                        iw = iw[:, None]
                        K_train = self.kernel.apply_to_dist(dist[np.ix_(train_i, train_i)], gamma=gamma)
                        K_train *= np.outer(iw[train_i], iw[train_i])
                        K_test = self.kernel.apply_to_dist(dist[np.ix_(test_i, train_i)], gamma=gamma)
                        for alpha_i, alpha in enumerate(self.krr_param_grid['alpha']):
                            if self.verbose > 0:
                                sys.stdout.write(',')
                                sys.stdout.flush()
                            K_train_ = K_train.copy()
                            K_train_.flat[::K_train_.shape[0] + 1] += alpha
                            try:
                                L_ = cholesky(K_train_, lower=True)
                                x = solve_triangular(L_, iw[train_i] * y_[train_i], lower=True)
                                dual_coef_ = iw[train_i] * solve_triangular(L_.T, x)
                                pred_mean = np.dot(K_test, dual_coef_)
                                if self.mae:
                                    e = np.mean(np.abs(pred_mean - y_[test_i]) * importance_weights[test_i, None]**2, 0)
                                else:
                                    e = np.mean(((pred_mean - y_[test_i]) ** 2) * importance_weights[test_i, None]**2, 0)
                            except np.linalg.LinAlgError:
                                e = np.inf
                            fold_errors[gamma_i, lamb_i, alpha_i] = e
                if self.verbose > 0:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                errors.append(fold_errors)
            errors = np.array(errors)
            errors = np.mean(errors, 0)  # average over folds

        self.dual_coefs_ = np.empty((y_.shape[1], X.shape[0]))
        self.alphas_ = np.empty(y_.shape[1])
        self.lambdas_ = np.empty(y_.shape[1])
        self.gammas_ = np.empty(y_.shape[1])
        if self.verbose > 0:
            elapsed = time.time() - t
            print('Refit [%dmin %dsec]' % (int(elapsed / 60),
                                           int(elapsed % 60)))
            sys.stdout.flush()
        print_count = 0

        if not self.single_combo:
            for i in range(y_.shape[1]):
                min_params = np.argsort(errors[:, :, :, i], axis=None)
                # lin_alg_errors = 0
                gamma_i, lamb_i, alpha_i = np.unravel_index(min_params[0],
                                                            errors.shape[:2])
                gamma = self.krr_param_grid['gamma'][gamma_i]
                lamb = self.krr_param_grid['lambda'][lamb_i]
                alpha = self.krr_param_grid['alpha'][alpha_i]
                self.alphas_[i] = alpha
                self.gammas_[i] = gamma
                self.lambdas_[i] = lamb

                if (gamma_i in (0, len(self.krr_param_grid['gamma']) - 1) or
                        lamb_i in (0, len(self.krr_param_grid['lambda']) - 1) or
                        alpha_i in (0, len(self.krr_param_grid['alpha']) - 1)):
                    if print_count <= 200:
                        fmtstr = '%d: gamma=%g\talpha=%g\tlambda=%g\terror=%g\tmean=%g'
                        print(fmtstr % (i, gamma, alpha, lamb,
                                        errors[gamma_i, lamb_i, alpha_i, i],
                                        errors[gamma_i, lamb_i, alpha_i, i] /
                                        np.mean(np.abs(y_[:, i]))))
                        print_count += 1
        else:
            errors = np.mean(errors, -1)  # average over outputs
            if self.verbose > 1:
                print('CV errors:')
                print(errors)
                print('Alpha params:')
                print(self.krr_param_grid['alpha'])
                print('Gamma params:')
                print(self.krr_param_grid['gamma'])
                print('Lambda params:')
                print(self.krr_param_grid['lambda'])
            if self.verbose > 0:
                print('Min error: ', np.min(errors))

            # print np.log(errors)
            # plt.imshow(np.log(errors))
            # plt.xticks(range(10), map('{:.1e}'.format, list(self.krr_param_grid['alpha'])))
            # plt.yticks(range(10), map('{:.1e}'.format, list(self.krr_param_grid['gamma'])))
            # plt.xlabel('alpha')
            # plt.ylabel('gamma')
            # plt.colorbar()
            # plt.show()
            min_params = np.argsort(errors, axis=None)
            if 'v' in self.krr_param_grid:
                v_i, gamma_i, lamb_i, alpha_i = np.unravel_index(min_params[0],
                                                                 errors.shape)
            else:
                gamma_i, lamb_i, alpha_i = np.unravel_index(min_params[0],
                                                            errors.shape)
            if 'v' in self.krr_param_grid:
                v = self.krr_param_grid['v'][v_i]
                print('v=', v)
            gamma = self.krr_param_grid['gamma'][gamma_i]
            alpha = self.krr_param_grid['alpha'][alpha_i]
            lamb = self.krr_param_grid['lambda'][lamb_i]

            if 'v' in self.krr_param_grid:
                if v == self.krr_param_grid['v'][0]:
                    print('v at lower edge.')
                if v == self.krr_param_grid['v'][-1]:
                    print('v at upper edge.')
            if len(self.krr_param_grid['gamma']) > 1:
                if gamma == self.krr_param_grid['gamma'][0]:
                    print('Gamma at lower edge.')
                if gamma == self.krr_param_grid['gamma'][-1]:
                    print('Gamma at upper edge.')
            if len(self.krr_param_grid['alpha']) > 1:
                if alpha == self.krr_param_grid['alpha'][0]:
                    print('Alpha at lower edge.')
                if alpha == self.krr_param_grid['alpha'][-1]:
                    print('Alpha at upper edge.')
            if len(self.krr_param_grid['lambda']) > 1:
                if lamb == self.krr_param_grid['lambda'][0]:
                    print('Lambda at lower edge.')
                if lamb == self.krr_param_grid['lambda'][-1]:
                    print('Lambda at upper edge.')
            self.alphas_[:] = alpha
            self.gammas_[:] = gamma
            self.lambdas_[:] = lamb

            if 'v' in self.krr_param_grid:
                alpha_add = get_alpha_add(self.n_basis, self.n_grid, self.delta, v)
                self.alphas_ *= alpha_add

        combos = list(zip(self.alphas_, self.gammas_, self.lambdas_))
        n_unique_combos = len(set(combos))
        self.L_fit_ = [None] * n_unique_combos
        for i, (alpha, gamma, lamb) in enumerate(set(combos)):
            if self.verbose > 0:
                elapsed = time.time() - t
                print('Parameter combinations ' +
                      '%d of %d [%dmin %dsec]' % (i + 1, n_unique_combos,
                                                  int(elapsed / 60),
                                                  int(elapsed % 60)))
                sys.stdout.flush()
            y_list = [i for i in range(y_.shape[1]) if
                      self.alphas_[i] == alpha and self.gammas_[i] == gamma and self.lambdas_[i] == lamb]

            iw = importance_weights**lamb
            iw = iw[:, None]
            K = self.kernel.apply_to_dist(dist, gamma=gamma)
            K *= np.outer(iw, iw)
            # np.exp(K, K)
            while True:
                K.flat[::K.shape[0] + 1] += alpha - (alpha / 10)
                try:
                    if self.verbose > 0:
                        print('trying cholesky decomposition, alpha', alpha)
                    L_ = cholesky(K, lower=True)
                    self.L_fit_[i] = L_
                    x = solve_triangular(L_, iw * y_[:, y_list], lower=True)
                    # x = solve_triangular(L_, y_[:, y_list], lower=True)
                    dual_coef_ = solve_triangular(L_.T, x)
                    self.dual_coefs_[y_list] = iw.T * dual_coef_.T.copy()
                    break
                except np.linalg.LinAlgError:
                    if self.verbose > 0:
                        print('LinalgError, increasing alpha')
                    alpha *= 10
                    self.alphas_[0] = alpha

        if self.copy_X:
            self.X_fit_ = X.copy()
            self.y_fit_ = y.copy()
        else:
            self.X_fit_ = X
            self.y_fit_ = y
        self.errors = errors

        if self.verbose > 0:
            elapsed = time.time() - t
            print('Done [%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))
            sys.stdout.flush()

    def add_sample(self, x, y):
        """ Adds a sample to the kernel matrix via an efficient update to the model
        Args:
            x   : The sample to be added
        """
        n = self.X_fit_.shape[0]
        print('n', n)
        if self.verbose > 1:
            print("adding training datapoint")
        self.X_fit_ = np.concatenate((self.X_fit_, x), axis=0)
        if self.verbose > 1:
            print("adding training label")
        self.y_fit_ = np.concatenate((self.y_fit_, y), axis=0)

        L_k = np.empty((self.L_fit_.shape[0], n + 1, n + 1))
        self.dual_coefs_ = np.empty((self.dual_coefs_.shape[0], n + 1))
        print(L_k.shape)
        for i, gamma in enumerate(np.unique(self.gammas_)):

            alpha = self.alphas_[i]

            if self.verbose > 1:
                print('Calculating kernel entries for new point')
            dist = euclidean_distances(x, self.X_fit_, squared=self.squared_dist)
            k = self.kernel.apply_to_dist(dist, gamma=gamma).T
            # print('n', n)

            k1 = k[:n]
            k2 = k[n:] + alpha

            if self.verbose > 1:
                print('Updating Cholesky factor')
            L_k[i, :n, :n] = self.L_fit_[i]
            L_k[i, :n, -1:] = 0
            L_k[i, -1:, :n] = solve_triangular(self.L_fit_[i], k1, lower=True).T
            # print('k2', k2)
            # print('dotprod', np.dot(L_k[i, -1:, :n], L_k[i, -1:, :n].T))
            # print('var', k2 - np.dot(L_k[i, -1:, :n], L_k[i, -1:, :n].T))
            L_k[i, -1:, -1:] = np.sqrt(k2 - np.dot(L_k[i, -1:, :n], L_k[i, -1:, :n].T))
            self.L_fit_ = L_k

            if self.verbose > 1:
                print('Updating dual_coefs')
            v = solve_triangular(L_k[i], self.y_fit_, lower=True)
            self.dual_coefs_[i] = solve_triangular(L_k[i].T, v).T

    def predict(self, X, verbose=None, variance=False, dist=None):
        t = time.time()

        if verbose is None:
            verbose = self.verbose

        y_ = np.empty(shape=(X.shape[0], len(self.alphas_)))
        if verbose > 0:
            elapsed = time.time() - t
            print('Computing distance matrix [%dmin %dsec]' % (
                int(elapsed / 60), int(elapsed % 60)))
            sys.stdout.flush()
        if dist is None:
            if X.shape == self.X_fit_.shape and np.allclose(X, self.X_fit_):
                dist = euclidean_distances(self.X_fit_, squared=self.squared_dist)
            else:
                dist = euclidean_distances(X, self.X_fit_, squared=self.squared_dist)
        if variance:
            if verbose > 0:
                elapsed = time.time() - t
                print('Test distances [%dmin %dsec]' % (int(elapsed / 60),
                                                        int(elapsed % 60)))
                sys.stdout.flush()
            dist_test = euclidean_distances(X, X, squared=self.squared_dist)
        pred_var = np.zeros((X.shape[0],))

        for i, gamma in enumerate(np.unique(self.gammas_)):
            if verbose > 0:
                print('Gamma %d of %d [%dmin %dsec]' % (i + 1,
                      len(np.unique(self.gammas_)), int(elapsed / 60),
                      int(elapsed % 60)))
                sys.stdout.flush()

            y_list = [i for i in range(len(self.gammas_)) if
                      self.gammas_[i] == gamma]
            K = self.kernel.apply_to_dist(dist, gamma=gamma)
            y_[:, y_list] = np.dot(K, self.dual_coefs_[y_list].T)
            if variance:
                K_test = self.kernel.apply_to_dist(dist_test, gamma=gamma)
                V = solve_triangular(self.L_fit_[i], K.T, lower=True)
                # v = np.dot(K, np.dot(self.L_fit_[i], K.T))
                v = np.sum(V * V, axis=0)
                pred_var = K_test.flat[::X.shape[0] + 1] - v

        if self.n_components is not None:
            y = self.pca.inverse_transform(y_)
        else:
            y = y_

        if y.shape[1] == 1:
            y = y.flatten()

        if verbose > 0:
            elapsed = time.time() - t
            print('Done [%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))
            sys.stdout.flush()

        if variance:
            return y, pred_var
        else:
            return y

    def save(self, filename):
        np.save(filename + '_alphas', self.alphas_)
        np.save(filename + '_dual_coefs', self.dual_coefs_)
        np.save(filename + '_gammas', self.gammas_)
        np.save(filename + '_lambdas', self.lambdas_)
        if not os.path.exists(filename + '_X_fit.npy') or self.replace_fit:
            np.save(filename + '_X_fit', self.X_fit_)
        np.save(filename + '_y_fit', self.y_fit_)
        # np.save(filename + '_L_fit', self.L_fit_)
        np.save(filename + '_errors', self.errors)
        np.save(filename + '_kernel', self.kernel)

    def load(self, filename):
        self.alphas_ = np.load(filename + '_alphas.npy', allow_pickle=True)
        self.dual_coefs_ = np.load(filename + '_dual_coefs.npy', allow_pickle=True)
        self.gammas_ = np.load(filename + '_gammas.npy', allow_pickle=True)
        self.X_fit_ = np.load(filename + '_X_fit.npy', allow_pickle=True)
        # self.L_fit_ = np.load(filename + '_L_fit.npy', allow_pickle=True)
        self.errors = np.load(filename + '_errors.npy', allow_pickle=True)
        self.kernel = np.load(filename + '_kernel.npy', allow_pickle=True)[()]
        if os.path.exists(filename + '_y_fit.npy'):
            self.y_fit_ = np.load(filename + '_y_fit.npy', allow_pickle=True)
        else:
            warnings.warn('No labels file found, not adding labels to model')
        if os.path.exists(filename + '_lambdas.npy'):
            self.lambdas_ = np.load(filename + '_lambdas.npy', allow_pickle=True)
        else:
            warnings.warn('No lambdas file found, not adding importance weights to model')
