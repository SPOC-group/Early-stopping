# Early stopping under label noise

Code and data for [*When is early stopping beneficial under label noise?*](https://arxiv.org/abs/2610.33846).
Use Python 3.10–3.12 and run the notebooks from this folder:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

- `Theory_main_figures.ipynb` recomputes Figures 2–5 from the analytical equations.
- `plotter.ipynb` reproduces Figure 1 and Figures 8–17 from `figures_data/FigureN.csv`. These CSVs contain the displayed means and uncertainty summaries.
- `Supplementary_Kernel_GF_RAF.ipynb` reproduces the kernel checks in Figures 6 and 7. Its default uses the stored simulation records in `kernel_data/` and recomputes the theoretical curves. Set `FORCE_RECOMPUTE_THEORY_VALIDATION = True` and `CIFAR_FORCE_RECOMPUTE = True` to run both simulations from scratch. CIFAR10 is downloaded from the official source when needed.

All plots are written to `Figures/`, including the manuscript filenames. The supplied data use the current paper's figure numbers. No training checkpoints are required.

## Neural-network experiments

The Python files include the data splits, architectures, losses, optimizers and training loops. Install the reference training versions:

```bash
python -m pip install tensorflow==2.18.0 keras==3.12.3 numpy==2.0.2 scipy==1.15.3
```

Pass `--cifar-dir` the extracted CIFAR10 Python-batch directory. Each call trains one seed and writes its measured trajectory; use `--help` for the options.

- `Figures1_8.py`: square loss, 7000 images, capacities `--kappa 1`, `10`, `100`; 1500 epochs, 30 planned seeds. RF-CNN trains only its readout, evaluated spectrally through `10**12` GD steps.
- `Figures9_10.py`: augmented cross-entropy, 6144 images, five seeds; 2000 epochs except WRN-28 at 1000.
- `Figure11.py`: six optimizer controls, five seeds; 800 epochs for CNNs and 400 for ResNet-16. `--summarize` combines completed seed CSVs.
- `Figures12_13.py`: ResNet sample-size, batch-size and random-label controls; 400 epochs with square loss or augmented cross-entropy. Use ten seeds at random-label fractions 0.1 and 0.2 for cross-entropy, five otherwise. The reported square-loss runs stop at 200 epochs for `n_alpha3` and 300 for `eps0p6`.
- `Figures14_17.py`: random labels on cats and dogs; ten seeds for square loss and five for cross-entropy. Square loss uses 1500 epochs. Cross-entropy horizons are 2000 for `cnn4` and `cnn6`, 1100 for `res16`, 900 for `res28`, and 1700 for `mobilev2`.

For example:

```bash
python Figures1_8.py --cifar-dir cifar-10-batches-py --architecture cnn4 --kappa 100 --seed 0 --epochs 1500 --output runs/square_seed0.csv
```

The architecture runners default to two epochs for a short check; set the desired horizon explicitly. Seed indices start at zero. Exact seeds, widths and learning rates are defined in the scripts. Repeated training can differ across software and hardware; the supplied aggregate CSVs reproduce the reported plots without retraining. Full kernel simulations and large-network training need substantially more time and memory than plotting.
