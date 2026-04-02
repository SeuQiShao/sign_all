# Predicting Dynamics of Ultra-Large Complex Systems by Inferring Governing Equations

## File Directory:
- **SIGN/SIGN_lasso_1d:** 1D dynamical inference as discussed in Section 2.2.
- **SIGN/SIGN_lasso_hd:** high-dimentional dynamical inference as discussed in Section 2.2.
- **SIGN/SIGN_lasso_\*d_noise/low_rank/miss_nodes/miss_edges:** Impact of robustnuess as discussed in Section 2.3.
- **SIGN/SIGN_true_hd_pred:** Simulate dynamics prediction experiment as discussed in Section 2.4.
- **SIGN/SIGN_true_pred:** Empirical East Pacific Ocean acoustic experiment as discussed in Section 2.5.

# Dependencies

The following are the required dependencies and their versions for the project:

- `torch==2.0.1`
- `torch-cluster==1.6.0`
- `torch-geometric==2.2.0`
- `torch-scatter==2.0.9`
- `torch-sparse==0.6.13`
- `torch-spline-conv==1.2.1`
- `numpy==1.21.5`
- `scikit-learn==1.0.2`

## Quick Start:
Navigate to any experimental folder and run:
```bash
bash run.sh
```
or
```
python trainer.py
```

## Generate Data:
In the /data directory, run:
```bash
bash data.sh
```
or use Python:
```bash
python generate_dataset.py
```
Note: Due to the large volume of simulation data, this repository does not include simulated data. ENSO data is located in the enso_data directory. Ensure that the data file paths are correctly modified. For efficiency, the simulation data generation steps are set to a maximum of 1000 steps for large networks. Reducing the sampling interval will help the model achieve more accurate inference results. Empirical network data can be downloaded from the [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/).
