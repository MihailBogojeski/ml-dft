import numpy as np
from scipy.spatial.distance import cdist
import ase
import ase.io
from sklearn.cluster import KMeans
import os
import subprocess
import re
from ml_dft.pykliep import DensityRatioEstimator

to_bohr = 1.889725989
to_angstrom = 0.529177249

number_to_symbol_map = {1: 'H', 6: 'C', 8: 'O'}
symbol_to_number_map = {'H': 1, 'C': 6, 'O': 8}


def angstrom_to_bohr(pos):
    return pos * to_bohr


def bohr_to_angstrom(pos):
    return pos * to_angstrom


def hartree_to_kcal(en):
    return en * 627.509


def eV_to_kcal(en):
    return en * 23.061


def hartree_to_eV(en):
    return en * 27.2116


def eV_to_hartree(en):
    return en / 27.2116


def kcal_to_hartree(en):
    return en / 627.509


def kcal_to_eV(en):
    return en / 23.061


def to_basis3d(X, n2, d=None):
    # Input:
    # X (n, d) matrix where the 3d axis are flattend into last axis
    # n2 is n/2, half the number of basis functions to use along one axis
    dx = d[0]
    dy = d[1]
    dz = d[2]
    n2x = n2[0]
    n2y = n2[1]
    n2z = n2[2]
    X2 = X.reshape(-1, dx, dy, dz)
    X2 = np.fft.rfft(X2, axis=-1)[:, :, :, :n2z]
    X2 = np.concatenate((X2.real, X2.imag), -1)
    X2 = np.fft.rfft(X2, axis=-2)[:, :, :n2y, :]
    X2 = np.concatenate((X2.real, X2.imag), -2)
    X2 = np.fft.rfft(X2, axis=-3)[:, :n2x, :, :]
    X2 = np.concatenate((X2.real, X2.imag), -3)
    return X2.reshape(X.shape[0], -1)


def rigid_transform_3D(atom_pos, base_pos):
    assert len(atom_pos) == len(base_pos)

    centroid_atom_pos = atom_pos.mean(axis=0)
    centroid_base_pos = base_pos.mean(axis=0)

    # centre the points
    atom_pos_cen = atom_pos - centroid_atom_pos
    base_pos_cen = base_pos - centroid_base_pos

    # dot is matrix multiplication for array
    H = base_pos_cen.T.dot(atom_pos_cen)

    U, S, Vt = np.linalg.svd(H)

    R = U.dot(Vt)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = U.dot(Vt)

    t = centroid_base_pos - R.dot(centroid_atom_pos)
    return R, t


def transform_molecule(A, base, heavy, return_rotation=False):
    """ Rotate molecule A so that its heavy atoms match base. """

    R, t = rigid_transform_3D(A[heavy], base[heavy])

    pos = A.dot(R.T) + t

    if len(heavy) == 3:
        # print(pos)
        pos -= pos[2, :]
        # print(pos)

        r1 = np.sqrt(np.sum((pos[0] - pos[2]) ** 2))
        r2 = np.sqrt(np.sum((pos[1] - pos[2]) ** 2))
        r3 = np.sqrt(np.sum((pos[0] - pos[1]) ** 2))
        ms = np.array([0, 0, -1])
        mr = np.sqrt(np.sum((ms - pos[2]) ** 2))
        mr1 = np.sqrt(np.sum((pos[0] - ms) ** 2))
        costheta = (r1 ** 2 + r2 ** 2 - r3 ** 2) / (2 * r1 * r2)
        cosm1 = (r1 ** 2 + mr ** 2 - mr1 ** 2) / (2 * r1 * mr)
        theta = np.arccos(costheta)
        theta1 = np.arccos(cosm1)

        rotheta = theta / 2 - theta1

        R_mat = [[1, 0, 0],
                 [0, np.cos(rotheta), -np.sin(rotheta)],
                 [0, np.sin(rotheta), np.cos(rotheta)]]

        R_mat = np.array(R_mat)

        pos = pos.dot(R_mat.T)
        pos += 10

    if return_rotation:
        return pos, R
    else:
        return pos


def reflect_through_plane(plane_points, ref_points):
    # create two vectors from plane points
    v1 = plane_points[0] - plane_points[1]
    v2 = plane_points[0] - plane_points[2]

    # calculate the normal vector by cross product
    normal_vec = np.cross(v1, v2)

    # calculate the plane constant as the dot product of the normal vector and a plane vector
    plane_const = np.sum(-plane_points[0] * normal_vec)

    # find value t, such that a line starting from a given point, moving in a perpendicular
    # direction to the plane, crosses that plane
    t = (-plane_const - np.dot(normal_vec, ref_points.T)) / np.dot(normal_vec, normal_vec)

    t = np.atleast_1d(t)
    t = t[:, np.newaxis]

    # to get reflected points, move points in the normal direction by 2t
    sym_points = ref_points + (2 * t * normal_vec)
    return sym_points


def normal_plane_on_line(plane_points, is_line_point):
    # separate points on line and other point
    line_points = plane_points[is_line_point, :]
    point = plane_points[np.logical_not(is_line_point), :]

    # calculate vector corresponding to line and another vector on the plane
    v1 = line_points[0, :] - line_points[1, :]
    v1 = v1[np.newaxis, :]
    v2 = line_points[0, :] - point

    # Calculate a vector normal to the plane from the two vectors
    normal_vec = np.cross(v1, v2)

    # Move a point on the line in the direction of the normal vector
    # to get a point on the normal plane
    normal_plane_point = line_points[0, :] + normal_vec

    # combine the point on the line with the new point to get three points
    # of the now normal plane
    normal_plane = np.concatenate((line_points, normal_plane_point), axis=0)

    return normal_plane


def symbols_to_numbers(symbols):
    numbers = []
    for s in symbols:
        numbers.append(symbol_to_number_map[s])

    return numbers


def numbers_to_symbols(numbers, join=True):
    symbols = []
    for n in numbers:
        symbols.append(number_to_symbol_map[n])

    if join:
        symbols = ''.join(symbols)

    return symbols


def calculate_potential(pos, charges, gaussian_width, grid, metric='sqeuclidean', verbose=1):
    if np.ndim(pos) == 2:
        pos = pos[np.newaxis, :]
    potentials = np.zeros((len(pos), len(grid)))

    for j in range(pos.shape[1]):
        if verbose > 0:
            print("Atom", j)
        atom_dist = cdist(pos[:, j, :], grid, metric=metric)
        potentials += charges[j] * np.exp(-atom_dist / (2 * (gaussian_width ** 2)))

    return potentials


def ase_to_npy(mols):
    arr = []
    for m in mols:
        arr.append(m.get_positions())

    return np.array(arr)


def npy_to_ase(arr, atom_list):
    mols = []
    for i in range(arr.shape[0]):
        mols.append(ase.Atoms(atom_list, positions=arr[i]))

    return mols


def energies_from_txt(filename, column, exclude_rows=0, energy_type='eV'):
    column -= 1

    with open(filename, 'r') as f:
        lines = list(f.readlines())
    lines = lines[exclude_rows:]
    words = [line.split() for line in lines]
    energies = [float(w[column]) for w in words]
    energies = np.array(energies)
    energies = np.reshape(energies, (-1, 1))
    if energy_type == 'hartree':
        energies = hartree_to_kcal(energies)
    elif energy_type == 'eV':
        energies = eV_to_kcal(energies)
    elif energy_type == 'kcal/mol':
        energies *= 1
    else:
        raise ValueError('energy type not supported for conversion')

    return energies


def positions_from_xyz(filename, convert_to_bohr=True):
    mols = ase.io.iread(filename)
    pos = ase_to_npy(mols)
    if convert_to_bohr:
        pos = angstrom_to_bohr(pos)

    return pos


def create_grid(grid_range):
    Y, X, Z = np.meshgrid(grid_range[0], grid_range[1], grid_range[2])

    X = X.flatten()[:, np.newaxis]
    Y = Y.flatten()[:, np.newaxis]
    Z = Z.flatten()[:, np.newaxis]
    grid = np.concatenate((X, Y, Z), axis=1)
    return grid


def get_molecule_dists(target, neighbours, charges=None):
    neighbour_dists = np.zeros((neighbours.shape[0], 1))
    for i in range(neighbours.shape[0]):
        atom_dists = np.sum((neighbours[i] - target)**2, axis=1)
        if charges is not None:
            atom_dists *= charges

        neighbour_dists[i] = np.sum(atom_dists)
    return neighbour_dists


def distance_nearest_neighbours(targets, neighbours, num_neighbours, charges=None):
    avg_distances = np.zeros((targets.shape[0], 1))
    for i in range(targets.shape[0]):
        neighbour_dists = get_molecule_dists(targets[i], neighbours, charges=charges)
        neighbour_dists.sort()
        avg_distances[i] = np.mean(neighbour_dists[:num_neighbours])

    return avg_distances


def load_KRR_model(run_id, dirname='.', kr_type='cc'):
    kernel = MultivariateGaussianProcessCV(krr_param_grid={"alpha": [1],
                                                           "gamma": [1]})
    kernel.load(os.path.join(dirname, kr_type + '_kr_' + str(run_id)))

    return kernel


def find_calculated_densities(folder_name):
    out = subprocess.check_output('ls -l ' + folder_name + '/*/density.cube', shell=True)
    out = str(out)
    out = out.split(r'\n')
    print(out)
    dens_ids = []
    for line in out:
        print(line)
        m = re.search('dft_calcs/(\d+)/', line)
        print(m)
        if m is not None:
            dens_ids.append(int(m.group(1)))

    print(dens_ids)
    print(len(dens_ids))
    return dens_ids


def k_means_sampling(num_samples, pos_file, random_seed=11):
    pos = np.load(pos_file)
    np.random.seed(random_seed)
    if not isinstance(num_samples, list):
        num_samples = [num_samples]
    inds = np.random.permutation(pos.shape[0])
    pos_shuff = pos[inds, :]

    pos_flat = np.reshape(pos_shuff, (pos_shuff.shape[0], -1))
    clusters = []
    for num_clust in num_samples:
        clusters.append(KMeans(n_clusters=num_clust).fit(pos_flat))

    sample_inds = []
    for clust in clusters:
        pos_dist = clust.transform(pos_flat)
        sample_inds.append(inds[np.argmin(pos_dist, axis=0)])

    return sample_inds


def align_to_base(pos, base_pos, heavy):
    pos_al = np.array(pos)
    for i in range(pos.shape[0]):
        pos_al[i] = transform_molecule(pos[i], base_pos, heavy)

    return pos_al


def str_to_atom_types(string):
    str_list = list(string)
    return np.array([str_list], dtype='<U1')


def estimate_importance_weights(train_set, test_set, samples, sigmas):
    kliep = DensityRatioEstimator(sigmas=sigmas)
    kliep.fit(train_set, test_set)
    weights = kliep.predict(samples)
    return weights
