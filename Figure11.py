#!/usr/bin/env python3
"""Optimizer controls for Figure 11.

The architectures, dataset, shuffle stream, six optimizers, rates and evaluation
schedule follow the original optimizer_robustness.py experiment. Shared model and
data definitions are imported from Figures1_8; no archived measurements are used
when training. Run one configuration at a time, then summarize its seed CSVs.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

PROTOCOLS = {
    "nesterov_b128": ("nesterov", 128, 1.0),
    "nesterov_b128_half": ("nesterov", 128, 0.5),
    "nesterov_b1024_fixed": ("nesterov", 1024, 1.0),
    "nesterov_b1024_sqrt": ("nesterov", 1024, math.sqrt(8)),
    "sgd_b128": ("sgd", 128, 5.0),
    "adam_b128": ("adam", 128, None),
}
ARCHITECTURES = ("two_layer", "shallow", "residual")
TIERS = ("interpolation", "high")
EXPECTED_DIGEST = "e94827f588241d144c6836ddc65b883ec4daa1a494745e733708ac0453d3365b"
EXPECTED_COUNTS = {
    ("interpolation", "two_layer"): 7124,
    ("interpolation", "shallow"): 7081,
    ("interpolation", "residual"): 7128,
    ("high", "two_layer"): 2746557,
    ("high", "shallow"): 2735965,
    ("high", "residual"): 2746577,
}


def configuration(tier="high", architecture="shallow", protocol="nesterov_b128"):
    if tier not in TIERS or architecture not in ARCHITECTURES:
        raise ValueError("Unknown tier or architecture")
    kind, batch, multiplier = PROTOCOLS[protocol]
    high = tier == "high"
    base = {"two_layer": .005 if high else .015,
            "shallow": .01 if high else .015,
            "residual": .01 if high else .02}[architecture]
    return dict(tier=tier, architecture=architecture, protocol=protocol,
                q=64 if high else 3, optimizer=kind, batch_size=batch,
                learning_rate=.001 if kind == "adam" else base * multiplier,
                epochs=400 if architecture == "residual" else 800,
                trainable_params=EXPECTED_COUNTS[tier, architecture])


def evaluation_epochs(final):
    points = {0, 1, 2, 3, 5, 8, 10, 15, 20, final}
    points.update(range(25, min(final, 100) + 1, 5))
    points.update(range(110, final + 1, 10))
    return sorted(point for point in points if 0 <= point <= final)


def run_trajectory(tier, architecture, protocol, seed, cifar_dir, output, epochs=None):
    if seed < 0 or (epochs is not None and epochs < 1):
        raise ValueError("Require a nonnegative seed and positive horizon")
    os.environ["CIFAR10_DIR"] = str(cifar_dir)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import tensorflow as tf
    import Figures1_8 as shared

    if tf.__version__ != "2.18.0":
        raise RuntimeError("Use TensorFlow 2.18.0 for the reference experiment")
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    task = configuration(tier, architecture, protocol)
    if epochs is not None:
        task["epochs"] = epochs
    split = shared.make_raf_split(shared.DataConfig(alpha=2.0))
    if split["index_digest"] != EXPECTED_DIGEST:
        raise RuntimeError("The original 6144-image dataset was not reproduced")
    model_seed = shared.INIT_SEED_BASE + 100000 * seed
    shared.set_all_seeds(model_seed)
    tf.keras.backend.clear_session()
    model = shared.build_model(shared.architecture_config(architecture, task["q"]), model_seed)
    if shared.count_params(model.trainable_variables) != task["trainable_params"]:
        raise RuntimeError("Architecture parameter count mismatch")
    if task["optimizer"] == "adam":
        opt = tf.keras.optimizers.Adam(task["learning_rate"], clipnorm=5.)
    else:
        nesterov = task["optimizer"] == "nesterov"
        opt = tf.keras.optimizers.SGD(task["learning_rate"],
                                      momentum=.9 if nesterov else 0.,
                                      nesterov=nesterov, clipnorm=5.)
    model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError())
    dataset = tf.data.Dataset.from_tensor_slices(
        (split["X_train"], split["y_train_pm1"][:, None]))
    dataset = dataset.shuffle(6144, seed=model_seed + 12345, reshuffle_each_iteration=True)
    dataset = dataset.batch(task["batch_size"], drop_remainder=False).prefetch(1)
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    wanted = set(evaluation_epochs(task["epochs"]))
    rows = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    class Recorder(tf.keras.callbacks.Callback):
        def record(self, epoch):
            scores = [np.asarray(self.model.predict(split[key], batch_size=256,
                      verbose=0), dtype=np.float64).reshape(-1)
                      for key in ("X_train", "X_test")]
            metrics = shared.metrics(*scores, split)
            rows.append(dict(**task, seed=seed, model_seed=model_seed, time=epoch,
                             updates=int(self.model.optimizer.iterations.numpy()),
                             index_digest=split["index_digest"], **metrics))
            temporary = output.with_suffix(output.suffix + ".tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            os.replace(temporary, output)

        def on_train_begin(self, logs=None):
            self.record(0)

        def on_epoch_end(self, epoch, logs=None):
            if epoch + 1 in wanted:
                self.record(epoch + 1)

    model.fit(dataset, epochs=task["epochs"], verbose=0,
              callbacks=[tf.keras.callbacks.TerminateOnNaN(), Recorder()])
    if [row["time"] for row in rows] != evaluation_epochs(task["epochs"]):
        raise RuntimeError("The run did not reach all prescribed checkpoints")
    return pd.DataFrame(rows)


def summarize(paths):
    """Mean per-run final-minus-best test error, with a Student-t 95% CI."""
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["tier", "architecture", "protocol", "seed"]
    if raw.duplicated(keys + ["time"]).any():
        raise ValueError("Duplicate checkpoints; pass each seed CSV only once")
    gaps = []
    for values, group in raw.groupby(keys, sort=True):
        group = group.sort_values("time")
        expected = configuration(*values[:3])["epochs"]
        if group.time.iloc[-1] != expected:
            raise ValueError(f"Incomplete Figure 11 run {values}: expected epoch {expected}")
        gaps.append(dict(zip(keys, values), gap_pp=100 *
                         (group.E_gen.iloc[-1] - group.E_gen.min())))
    rows = []
    for values, group in pd.DataFrame(gaps).groupby(keys[:3], sort=True):
        n = len(group)
        if n < 2:
            raise ValueError("At least two seeds are required for a confidence interval")
        rows.append(dict(zip(keys[:3], values), gap_pp=group.gap_pp.mean(),
                         ci95_halfwidth_pp=student_t.ppf(.975, n-1) *
                         group.gap_pp.std(ddof=1) / np.sqrt(n)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=TIERS, default="high")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="shallow")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="nesterov_b128")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="Short smoke-run override; omit for the paper horizon")
    parser.add_argument("--cifar-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("trajectory.csv"))
    parser.add_argument("--summarize", type=Path, nargs="+", metavar="CSV",
                        help="Aggregate completed seed runs instead of training")
    args = parser.parse_args()
    if args.summarize:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summarize(args.summarize).to_csv(args.output, index=False)
    else:
        if args.cifar_dir is None:
            parser.error("--cifar-dir is required for training")
        run_trajectory(args.tier, args.architecture, args.protocol, args.seed,
                       args.cifar_dir, args.output, args.epochs)
    print(args.output)


if __name__ == "__main__":
    main()
