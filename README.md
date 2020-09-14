ML-DFT: Machine learning for density functional approximations
This repository contains the implementation for the kernel ridge regression based density functional approximation method described in the paper "Quantum chemical accuracy from density functionalapproximations via machine learning".

##### Requirements:
- python==3.7
- numpy>=1.19
- scikit-learn>=0.23
- scipy>=1.5
- sacred==0.8
- ase>=3.20

##### Installation:
Clone the repository by running:
```
git clone https://github.com/MihailBogojeski/ml-dft.git
```

To install package along with all requirements, simply run:
```
cd ml-dft
pip install -r requirements.txt
pip install .
```

##### Training and testing a basic model
To train and test an example model based on a small dataset consisting of 102 water geometries, use the following command:
```
python run_ml_dft_experiment.py water_102 water_102 50 50
```
