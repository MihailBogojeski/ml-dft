import subprocess
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run ML-DFT experiment using "sacred" framework')
parser.add_argument('train_folder', type=str, help='Folder where the training data is contained')
parser.add_argument('test_folder', type=str, help='Folder where the test data is contained')
parser.add_argument('n_training', type=int, help='Number of training samples used')
parser.add_argument('n_test', type=int, help='Number of test samples used')
parser.add_argument('--run_id', type=int, default=1, help='ID of the experiment, used to differentiate filenames')
parser.add_argument('--gaussian_width', type=float, default=0.6, help='Width parameter used for the artificial gaussian potentials')
parser.add_argument('--spacing', type=float, default=0.3, help='Spacing parameter used to sample the artificial gaussian potentials grid')

args = parser.parse_args()
# Comment this out after you've calcualted the potentials once so it's not needlessly repeated
# Calculate the artificial potentials
subprocess.call('python ml_dft/dft_ml.py calculate_potentials with "train_dir=' + args.train_folder + '" "test_dir=' + args.test_folder +
                '" run_id=' + str(args.run_id) +
                ' "gaussian_width=' + str(args.gaussian_width) + '" "grid_spacing=' + str(args.spacing) +
                '" grid_file=""', shell=True)

# Train and test the full model on the training data

if args.train_folder != args.test_folder:
    train_inds = np.arange(args.n_training)
    test_inds = np.arange(args.n_test)
    np.save('train_inds_file.npy', train_inds)
    np.save('test_inds_file.npy', test_inds)
else:
    inds = list(range(args.n_training + args.n_test))
    train_inds = np.random.choice(inds, args.n_training, replace=False)
    test_inds = np.setdiff1d(inds, train_inds)
    np.save('train_inds_file.npy', train_inds)
    np.save('test_inds_file.npy', test_inds)


# # Train full ML-HK model
subprocess.call('python ml_dft/dft_ml.py with "train_dir=' + args.train_folder + '" "test_dir=' + args.test_folder +
                '" "train_inds_file=train_inds_file.npy" "test_inds_file=test_inds_file.npy' +
                '" "energy_type=dft" "use_true_densities=False" "run_id=' + str(args.run_id) +
                '" "gaussian_width=' + str(args.gaussian_width) + '" "grid_spacing=' + str(args.spacing) +
                '"', shell=True)
