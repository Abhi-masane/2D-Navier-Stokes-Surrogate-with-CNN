# 2D Navier-Stokes surrogate with a CNN

This project trains a small U-Net to approximate steady 2D flow around a cylinder. The CFD data is generated with OpenFOAM for different Reynolds numbers and then sampled onto a fixed Cartesian grid for training in PyTorch.

The network takes two input channels:

- the cylinder/fluid mask
- a constant map containing the normalised Reynolds number

It predicts three output fields: `ux`, `uy`, and pressure `p`.

## Repository layout

```text
baseCase/           OpenFOAM case used as the template
scripts/            data preparation and small helper scripts
src/                PyTorch dataset, model, training and evaluation code
dataset/            converted .npz cases and manifest.json
results/            saved figures and model checkpoint after training
make_cases.sh       creates one OpenFOAM case per Reynolds number
run_all.sh          runs simpleFoam and foamToVTK for all cases
```

## Data generation

The Reynolds number is varied from 5 to 40. The kinematic viscosity is fixed at `0.01` and the cylinder diameter is `1.0`, so the inlet velocity is calculated from

```text
U = Re * nu / D
```

To create and run the OpenFOAM cases:

```bash
./make_cases.sh
./run_all.sh
```

The converted fields are sampled on a `129 x 65` grid covering `x = [-5, 15]` and `y = [-5, 5]`.

Run the conversion from the repository root:

```bash
python3 scripts/convert_all.py
```

Velocity is stored as `u / U_in` and pressure as `p / U_in^2`. Grid points inside the cylinder are removed from the loss with a binary mask.

## Training

The model is a compact U-Net with two downsampling stages. Training uses Adam and mean squared error on the masked flow field.

```bash
cd src
python3 train.py
```

The checkpoint with the lowest validation loss is saved as `results/best.pt`. Model weights are ignored by Git, so a fresh clone needs to be trained before running the evaluation unless a checkpoint is supplied separately.

## Evaluation

From the `src` directory:

```bash
python3 evaluate.py
```

The script reports relative L2 errors for velocity and pressure and writes a comparison plot to `results/test_case.png`.

For the run used for the figures currently in this repository, the reported test errors were approximately:

- velocity: 4.05%
- pressure: 13.85%

![Prediction example](results/test_case.png)
