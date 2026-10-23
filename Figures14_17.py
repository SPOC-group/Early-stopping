"""Cat/dog random-label experiments for Figures 14--17."""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def backend(recipe):
    if recipe == 'square':
        import Figures1_8 as module
    elif recipe == 'bce':
        import Figures9_10 as module
    else:
        raise ValueError('Choose square or bce')
    return module


def select_indices(train_labels, test_labels, module, cfg):
    counts = module.raf_counts(cfg, 3072)
    rng = np.random.default_rng(cfg.data_seed)
    pos, neg, auto = [np.flatnonzero(train_labels == i) for i in (3, 5, 1)]
    choose = lambda a, n: module.sample_without_replacement(a, n, rng)
    clean_pos = choose(pos, counts['n_struct_pos'])
    clean_neg = choose(neg, counts['n_struct_neg'])
    # Retain the original RNG stream, clean/validation/test IDs and assigned random labels.
    choose(auto, counts['n_mem'])
    test_pos = choose(np.flatnonzero(test_labels == 3), counts['n_test_pos'])
    test_neg = choose(np.flatnonzero(test_labels == 5), counts['n_test_neg'])
    random_labels = rng.choice([-1., 1.], counts['n_mem']).astype('float32')
    val_pos = choose(np.setdiff1d(pos, clean_pos), counts['n_val_pos'])
    val_neg = choose(np.setdiff1d(neg, clean_neg), counts['n_val_neg'])
    remaining_pos = np.setdiff1d(pos, np.r_[clean_pos, val_pos])
    remaining_neg = np.setdiff1d(neg, np.r_[clean_neg, val_neg])
    replace_rng = np.random.default_rng(cfg.data_seed + 928173)
    n_pos = counts['n_mem'] // 2
    mem = np.r_[replace_rng.choice(remaining_pos, n_pos, replace=False),
                replace_rng.choice(remaining_neg, counts['n_mem'] - n_pos, replace=False)]
    train = np.r_[clean_pos, clean_neg, mem]
    val = np.r_[val_pos, val_neg]
    if len(np.unique(train)) != counts['n_total'] or np.intersect1d(train, val).size:
        raise RuntimeError('Overlapping clean/random/validation images')
    return dict(train_ids=train, val_ids=val, test_ids=np.r_[test_pos, test_neg],
                random_labels=random_labels, counts=counts)


def load_data(recipe='square'):
    module = backend(recipe)
    cfg = module.DataConfig(alpha=module.N_TOTAL/3072, eps=.2, data_seed=0)
    x, y, xt, yt = module.load_cifar10()
    ids = select_indices(y, yt, module, cfg)
    counts = ids['counts']
    raw = x[ids['train_ids']]
    mean, std = module.fit_zscore(raw.reshape(len(raw), -1).astype('float32'))
    convert = lambda images: module._apply_zscore_images(images, mean, std)
    return dict(
        X_train=convert(raw),
        y_train_pm1=np.r_[np.ones(counts['n_struct_pos']),
                         -np.ones(counts['n_struct_neg']), ids['random_labels']].astype('float32'),
        mem_mask=np.arange(counts['n_total']) >= counts['n_struct'],
        X_test=convert(xt[ids['test_ids']]),
        y_test_pm1=np.where(yt[ids['test_ids']] == 3, 1., -1.).astype('float32'),
        X_val=convert(x[ids['val_ids']]),
        y_val_pm1=np.where(y[ids['val_ids']] == 3, 1., -1.).astype('float32'),
        train_ids=ids['train_ids'], val_ids=ids['val_ids'], test_ids=ids['test_ids'],
        true_train_labels=y[ids['train_ids']], mean=mean, std=std)


def run_trajectory(recipe='square', architecture='cnn4', epochs=2, seed=0,
                   kappa=100, split=None):
    module = backend(recipe)
    split = load_data(recipe) if split is None else split
    return module.run_trajectory(architecture, epochs, seed, kappa, split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=['square', 'bce'], default='square')
    parser.add_argument('--architecture', default='cnn4')
    parser.add_argument('--kappa', type=int, choices=[1, 10, 100], default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--cifar-dir', required=True)
    parser.add_argument('--output', type=Path, default=Path('trajectory.csv'))
    args = parser.parse_args()
    os.environ['CIFAR10_DIR'] = args.cifar_dir
    backend(args.recipe).configure_runtime()
    frame = run_trajectory(args.recipe, args.architecture, args.epochs, args.seed, args.kappa)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
