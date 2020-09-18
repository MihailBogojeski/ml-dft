ML-DFT: Machine learning for density functional approximations
This repository contains the implementation for the kernel ridge regression based density functional approximation method described in the paper "Quantum chemical accuracy from density functional approximations via machine learning".

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
Install time should range from 3-10 minutes, depending on internet speed.

##### Training and testing a basic model
To train and test an example model based on a small dataset consisting of 102 water geometries, use the following command:
```
python run_ml_dft_experiment.py water_102 water_102 50 50
```
The training and evalutaion of this example model should be done in under a minute.
The final lines of the expected output should look like:
```
Correlation:  0.9999208434612572
RMSE:  0.2636152741637275
MAE:  0.13975325554005394
Max:  1.4237659260907094
INFO - Machine learning DFT - Result: 0.13975325554005394
INFO - Machine learning DFT - Completed after 0:00:15
```
Since the crossvalidation process is random the values can be different, however if the model is trained successfully the MAE should be in the range from 0.2 to 0.06.
