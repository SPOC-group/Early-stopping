#!/usr/bin/env python3
"""Square loss with constant-rate Nesterov SGD; no augmentation.
Example: python Figures1_8.py --cifar-dir /path/to/cifar-10-batches-py --epochs 2
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.linalg import eigh, svd
from scipy.stats import t as student_t

INIT_SEED_BASE = 24681357
Q_VALUES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64)
ARCHITECTURES = ("frozen", "mlp", "two_layer", "shallow", "residual")
METRICS = ("E_gen", "E_mem", "E_train", "E_rule_train", "train_loss", "test_loss")
DISPLAY_WINDOW = 7
N_TOTAL = 7000
ARCHS = ("rf_readout", "cnn2", "cnn4", "cnn6", "res16", "res28")
MAIN_ARCHS = ("rf_readout", "cnn2", "cnn4", "res28")
LABELS = dict(
    rf_readout="RF-CNN (readout matched)",
    cnn2="Two-layer CNN",
    cnn4="Four-layer CNN",
    cnn6="Six-layer CNN",
    res16="ResNet-16",
    res28="WRN-28",
    mobilev2="MobileNetV2",
)
COLORS = dict(
    rf_readout="#2875B9",
    cnn2="#009B83",
    cnn4="#D84343",
    cnn6="#C28B18",
    res16="#242424",
    res28="#AD4B9C",
    mobilev2="#2875B9",
)


@dataclass(frozen=True)
class DataConfig:
    data_seed: int = 0
    alpha: float = 2.0
    eps: float = 0.2
    n_test: int = 2000
    n_val: int = 1000
    class_pos: int = 3
    class_neg: int = 5
    class_mem: int = 1
    preprocess_mode: str = "zscore"


DATA_CFG = DataConfig()


def find_cifar10_dir() -> Path:
    candidates: list[Path] = []
    if os.environ.get("CIFAR10_DIR"):
        candidates.append(Path(os.environ["CIFAR10_DIR"]))
    keras_home = Path(os.environ.get("KERAS_HOME", Path.home() / ".keras"))
    candidates.extend(
        [
            keras_home
            / "datasets"
            / "cifar-10-batches-py-target"
            / "cifar-10-batches-py",
            Path.cwd()
            / ".keras"
            / "datasets"
            / "cifar-10-batches-py-target"
            / "cifar-10-batches-py",
            Path.home()
            / ".keras"
            / "datasets"
            / "cifar-10-batches-py-target"
            / "cifar-10-batches-py",
        ]
    )
    required = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
    for base in candidates:
        if all(((base / name).is_file() for name in required)):
            return base.resolve()
    raise FileNotFoundError(
        "CIFAR10 is unavailable. Set CIFAR10_DIR to the extracted directory containing data_batch_1,...,data_batch_5,test_batch,batches.meta."
    )


def load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle, encoding="latin1")
    except Exception as exc:
        raise RuntimeError(
            f"Could not read CIFAR batch {path}; the cache may be corrupted."
        ) from exc
    data = np.asarray(obj["data"], dtype=np.uint8)
    labels = np.asarray(obj["labels"], dtype=np.int64)
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return (images, labels)


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = find_cifar10_dir()
    train_batches = [load_cifar_batch(base / f"data_batch_{i}") for i in range(1, 6)]
    X_train = np.concatenate([batch[0] for batch in train_batches], axis=0)
    y_train = np.concatenate([batch[1] for batch in train_batches], axis=0)
    (X_test, y_test) = load_cifar_batch(base / "test_batch")
    if X_train.shape != (50000, 32, 32, 3) or X_test.shape != (10000, 32, 32, 3):
        raise RuntimeError(
            f"Unexpected CIFAR10 shapes: train={X_train.shape}, test={X_test.shape}"
        )
    return (X_train, y_train, X_test, y_test)


def sample_without_replacement(
    indices: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if n < 0 or len(indices) < n:
        raise ValueError(f"Cannot draw {n} elements from {len(indices)} indices.")
    return rng.choice(indices, size=n, replace=False)


def raf_counts(cfg: DataConfig, d: int) -> dict:
    n_total = int(round(cfg.alpha * d))
    n_struct = int(2 * round((1.0 - cfg.eps) * n_total / 2.0))
    n_struct = min(max(n_struct, 2), n_total - 1)
    if n_struct % 2:
        n_struct += 1 if n_struct < n_total else -1
    if cfg.n_test % 2 or cfg.n_val % 2:
        raise ValueError("n_test and n_val must be even.")
    n_mem = n_total - n_struct
    return {
        "d": int(d),
        "n_total": int(n_total),
        "n_struct": int(n_struct),
        "n_mem": int(n_mem),
        "n_struct_pos": int(n_struct // 2),
        "n_struct_neg": int(n_struct // 2),
        "n_test_pos": int(cfg.n_test // 2),
        "n_test_neg": int(cfg.n_test // 2),
        "n_val_pos": int(cfg.n_val // 2),
        "n_val_neg": int(cfg.n_val // 2),
        "realized_alpha": float(n_total / d),
        "realized_eps": float(n_mem / n_total),
    }


def fit_zscore(X_train_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train_flat.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_train_flat.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-08] = 1.0
    return (mean, std)


def _apply_zscore_images(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    flat = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    flat = (flat - mean) / std
    return flat.reshape(-1, 32, 32, 3).astype(np.float32)


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def make_raf_split(cfg: DataConfig = DATA_CFG) -> dict:
    
    (X_train_full, y_train_full, X_test_full, y_test_full) = load_cifar10()
    rng = np.random.default_rng(cfg.data_seed)
    d = int(np.prod(X_train_full.shape[1:]))
    counts = raf_counts(cfg, d)
    idx_train_pos = np.flatnonzero(y_train_full == cfg.class_pos)
    idx_train_neg = np.flatnonzero(y_train_full == cfg.class_neg)
    idx_train_mem = np.flatnonzero(y_train_full == cfg.class_mem)
    idx_test_pos = np.flatnonzero(y_test_full == cfg.class_pos)
    idx_test_neg = np.flatnonzero(y_test_full == cfg.class_neg)
    sel_train_pos = sample_without_replacement(
        idx_train_pos, counts["n_struct_pos"], rng
    )
    sel_train_neg = sample_without_replacement(
        idx_train_neg, counts["n_struct_neg"], rng
    )
    sel_train_mem = sample_without_replacement(idx_train_mem, counts["n_mem"], rng)
    sel_test_pos = sample_without_replacement(idx_test_pos, counts["n_test_pos"], rng)
    sel_test_neg = sample_without_replacement(idx_test_neg, counts["n_test_neg"], rng)
    y_train_pm1 = np.concatenate(
        [
            np.ones(counts["n_struct_pos"], dtype=np.float32),
            -np.ones(counts["n_struct_neg"], dtype=np.float32),
            rng.choice([-1.0, 1.0], size=counts["n_mem"], replace=True).astype(
                np.float32
            ),
        ]
    )
    remaining_pos = np.setdiff1d(idx_train_pos, sel_train_pos, assume_unique=False)
    remaining_neg = np.setdiff1d(idx_train_neg, sel_train_neg, assume_unique=False)
    sel_val_pos = sample_without_replacement(remaining_pos, counts["n_val_pos"], rng)
    sel_val_neg = sample_without_replacement(remaining_neg, counts["n_val_neg"], rng)
    X_train_raw = np.concatenate(
        [
            X_train_full[sel_train_pos],
            X_train_full[sel_train_neg],
            X_train_full[sel_train_mem],
        ],
        axis=0,
    )
    X_val_raw = np.concatenate(
        [X_train_full[sel_val_pos], X_train_full[sel_val_neg]], axis=0
    )
    X_test_raw = np.concatenate(
        [X_test_full[sel_test_pos], X_test_full[sel_test_neg]], axis=0
    )
    y_val_pm1 = np.concatenate(
        [
            np.ones(counts["n_val_pos"], dtype=np.float32),
            -np.ones(counts["n_val_neg"], dtype=np.float32),
        ]
    )
    y_test_pm1 = np.concatenate(
        [
            np.ones(counts["n_test_pos"], dtype=np.float32),
            -np.ones(counts["n_test_neg"], dtype=np.float32),
        ]
    )
    train_flat = X_train_raw.reshape(len(X_train_raw), -1).astype(np.float32)
    (mean, std) = fit_zscore(train_flat)
    X_train = _apply_zscore_images(X_train_raw, mean, std)
    X_val = _apply_zscore_images(X_val_raw, mean, std)
    X_test = _apply_zscore_images(X_test_raw, mean, std)
    mem_mask = np.zeros(counts["n_total"], dtype=bool)
    mem_mask[counts["n_struct"] :] = True
    index_digest = _array_digest(
        sel_train_pos,
        sel_train_neg,
        sel_train_mem,
        sel_test_pos,
        sel_test_neg,
        y_train_pm1,
    )
    return {
        "X_train": X_train,
        "y_train_pm1": y_train_pm1,
        "X_val": X_val,
        "y_val_pm1": y_val_pm1,
        "X_test": X_test,
        "y_test_pm1": y_test_pm1,
        "mem_mask": mem_mask,
        "counts": counts,
        "index_digest": index_digest,
        "data_config": asdict(cfg),
    }


@dataclass(frozen=True)
class ArchitectureConfig:
    q: int
    architecture: str
    target_params: int
    mlp_width: int = 0
    c1: int = 0
    c2: int = 0
    shallow_channels: tuple[int, int, int, int] = (0, 0, 0, 0)
    residual_base: int = 0


@dataclass(frozen=True)
class Task:
    task_id: int
    seed_index: int
    loss: str
    q: int
    architecture: str
    spec_id: str


def require_tensorflow() -> None:
    if tf is None:
        raise RuntimeError(
            "TensorFlow is required for run/smoke."
        )


def residual_parameter_count(q: int) -> int:
    projection = 0 if q == 16 else 16 * q
    return 667 * q * q + 204 * q + 465 + projection


def two_layer_parameter_count(c1: int, c2: int) -> int:
    return 27 * c1 + c1 * c2 + c2 + 1


def shallow_parameter_count(channels: tuple[int, int, int, int]) -> int:
    (a, b, c, d) = channels
    return (
        27 * a + 2 * a + 9 * a * b + 2 * b + 9 * b * c + 2 * c + 9 * c * d + 5 * d + 1
    )


def mlp_parameter_count(width: int) -> int:
    return 194 * width + 1


def architecture_config(architecture: str, q: int) -> ArchitectureConfig:
    if q not in Q_VALUES:
        raise ValueError(f"Unsupported capacity index q={q}")
    if architecture not in ARCHITECTURES:
        raise ValueError(architecture)
    target = residual_parameter_count(q)
    if q == 1:
        mlp_width = 7
        (c1, c2) = (3, 318)
        shallow = (2, 4, 8, 12)
    elif q == 3:
        mlp_width = 37
        (c1, c2) = (9, 688)
        shallow = (5, 10, 20, 25)
    elif q == 64:
        mlp_width = 14103
        (c1, c2) = (192, 14204)
        shallow = (108, 216, 432, 432)
    else:
        mlp_width = max(1, int(round((target - 1) / 194)))
        c1 = 3 * q
        c2 = max(1, int(round((target - 27 * c1 - 1) / (c1 + 1))))
        base = max(1, int(round(math.sqrt(target / 234.0))))
        (a, b, c) = (base, 2 * base, 4 * base)
        count_without_last = shallow_parameter_count((a, b, c, 0))
        last = max(1, int(round((target - count_without_last) / (9 * c + 5))))
        shallow = (a, b, c, last)
    return ArchitectureConfig(
        q=q,
        architecture=architecture,
        target_params=target,
        mlp_width=mlp_width,
        c1=c1,
        c2=c2,
        shallow_channels=shallow,
        residual_base=q,
    )


def expected_trainable_params(config: ArchitectureConfig) -> int:
    if config.architecture == "mlp":
        return mlp_parameter_count(config.mlp_width)
    if config.architecture in {"frozen", "two_layer"}:
        if config.architecture == "frozen":
            return config.c2 + 1
        return two_layer_parameter_count(config.c1, config.c2)
    if config.architecture == "shallow":
        return shallow_parameter_count(config.shallow_channels)
    return residual_parameter_count(config.residual_base)


def count_params(variables) -> int:
    return int(sum((np.prod(variable.shape) for variable in variables)))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    require_tensorflow()
    tf.keras.utils.set_random_seed(seed)


def seeded_conv(filters: int, kernel: int, stride: int, seed: int, name: str):
    return tf.keras.layers.Conv2D(
        filters,
        kernel,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
        name=name,
    )


def relu(x, name: str):
    return tf.keras.layers.ReLU(name=name)(x)


def legacy_block(x, filters: int, stride: int, seed: int, name: str):
    x = seeded_conv(filters, 3, stride, seed, f"{name}_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return relu(x, f"{name}_relu")


def wrn_block(x, filters: int, stride: int, seed: int, name: str):
    shortcut = x
    y = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    y = relu(y, f"{name}_relu1")
    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = seeded_conv(filters, 1, stride, seed + 1, f"{name}_shortcut")(y)
    y = seeded_conv(filters, 3, stride, seed + 2, f"{name}_conv1")(y)
    y = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = relu(y, f"{name}_relu2")
    y = seeded_conv(filters, 3, 1, seed + 3, f"{name}_conv2")(y)
    return tf.keras.layers.Add(name=f"{name}_add")([shortcut, y])


def two_layer_features(config: ArchitectureConfig, seed: int):
    inputs = tf.keras.layers.Input((32, 32, 3), name="image")
    x = tf.keras.layers.AveragePooling2D(8, name="pool_to_4x4")(inputs)
    x = tf.keras.layers.Conv2D(
        config.c1,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + 1),
        name="conv3x3",
    )(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="spatial_average")(x)
    x = tf.keras.layers.Reshape((1, 1, config.c1), name="as_1x1")(x)
    x = tf.keras.layers.Conv2D(
        config.c2,
        1,
        padding="valid",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + 2),
        name="conv1x1",
    )(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    features = tf.keras.layers.Flatten(name="features")(x)
    return tf.keras.Model(inputs, features, name=f"TwoLayerFeatures_q{config.q}")


def build_model(config: ArchitectureConfig, seed: int):
    architecture = config.architecture
    if architecture == "mlp":
        inputs = tf.keras.layers.Input((32, 32, 3), name="image")
        x = tf.keras.layers.AveragePooling2D(4, name="pool_to_8x8")(inputs)
        x = tf.keras.layers.Flatten(name="flatten_8x8")(x)
        x = tf.keras.layers.Dense(
            config.mlp_width,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + 1),
            name="hidden",
        )(x)
        x = tf.keras.layers.ReLU(name="hidden_relu")(x)
        output = tf.keras.layers.Dense(
            1,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=1.0 / math.sqrt(config.mlp_width), seed=seed + 2
            ),
            name="score",
        )(x)
        model = tf.keras.Model(inputs, output, name=f"MLP_q{config.q}")
    elif architecture == "two_layer":
        features = two_layer_features(config, seed)
        output = tf.keras.layers.Dense(
            1,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=1.0 / math.sqrt(config.c2), seed=seed + 3
            ),
            name="score",
        )(features.output)
        model = tf.keras.Model(features.input, output, name=f"TwoLayerCNN_q{config.q}")
    elif architecture == "shallow":
        inputs = tf.keras.layers.Input((32, 32, 3), name="image")
        x = inputs
        for index, (filters, stride) in enumerate(
            zip(config.shallow_channels, (1, 2, 2, 1)), start=1
        ):
            x = legacy_block(x, filters, stride, seed + index, f"block{index}")
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
        x = tf.keras.layers.LayerNormalization(name="feature_layer_norm")(x)
        output = tf.keras.layers.Dense(
            1,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + 5),
            name="score",
        )(x)
        model = tf.keras.Model(inputs, output, name=f"FourLayerCNN_q{config.q}")
    elif architecture == "residual":
        q = config.residual_base
        inputs = tf.keras.layers.Input((32, 32, 3), name="image")
        x = seeded_conv(16, 3, 1, seed + 1, "stem_conv")(inputs)
        layer_seed = seed + 10
        for group_index, filters in enumerate((q, 2 * q, 4 * q)):
            for block_index in range(2):
                stride = 1 if group_index == 0 or block_index > 0 else 2
                x = wrn_block(
                    x,
                    filters,
                    stride,
                    layer_seed,
                    f"g{group_index + 1}_b{block_index + 1}",
                )
                layer_seed += 10
        x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
        x = tf.keras.layers.ReLU(name="head_relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
        output = tf.keras.layers.Dense(
            1,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + 100),
            name="score",
        )(x)
        model = tf.keras.Model(inputs, output, name=f"ResNet16_q{q}")
    else:
        raise ValueError(architecture)
    expected = expected_trainable_params(config)
    actual = count_params(model.trainable_variables)
    if actual != expected:
        raise RuntimeError(
            f"Parameter mismatch for {architecture}/q={config.q}: {actual} != {expected}"
        )
    return model


def legacy_square_rate(task: Task) -> float:
    q = task.q
    architecture = task.architecture
    if architecture == "mlp":
        return 0.05 if q <= 3 else 0.02
    if architecture == "two_layer":
        if q <= 1:
            return 0.02
        if q <= 4:
            return 0.015
        if q <= 16:
            return 0.01
        return 0.005
    if architecture == "shallow":
        if q <= 1:
            return 0.02
        if q <= 4:
            return 0.015
        return 0.01
    if architecture == "residual":
        return 0.02 if q <= 8 else 0.01
    return 0.01


def residual_count(q, blocks=2):
    return residual_parameter_count(q) + (blocks - 2) * (378 * q * q + 28 * q)


def convolution_count(channels):
    (previous, total) = (3, 0)
    for width in channels:
        total += 9 * previous * width + 2 * width
        previous = width
    return total + 3 * channels[-1] + 1


def channels_for(target, depth):
    if depth == 4:
        b = max(1, round(math.sqrt(target / 234.0)))
        prefix = [b, 2 * b, 4 * b]
    else:
        b = max(1, round(math.sqrt(target / 522.0)))
        prefix = [b, 2 * b, 4 * b, 4 * b, 4 * b]
    before = convolution_count(prefix + [0]) - 1
    last = max(1, round((target - before - 1) / (9 * prefix[-1] + 5)))
    return prefix + [last]


def legacy_configuration(architecture, kappa, n):
    target = n * kappa
    q = min(range(1, 256), key=lambda w: (abs(residual_count(w) - target), w))
    result = dict(architecture=architecture, kappa=kappa, target_params=target, q=q)
    if architecture in ("cnn2", "rf_readout"):
        c1 = 3 * q
        c2 = (
            target - 1
            if architecture == "rf_readout"
            else max(1, round((target - 27 * c1 - 1) / (c1 + 1)))
        )
        p = two_layer_parameter_count(c1, c2)
        result.update(
            c1=c1,
            c2=c2,
            architecture_params=p,
            readout_params=c2 + 1,
            trainable_params=c2 + 1 if architecture == "rf_readout" else p,
        )
        rate = legacy_square_rate(Task(0, 0, "square", q, "two_layer", ""))
    elif architecture in ("cnn4", "cnn6"):
        channels = channels_for(target, int(architecture[-1]))
        p = convolution_count(channels)
        result.update(
            channels=channels,
            architecture_params=p,
            trainable_params=p,
            readout_params=channels[-1] + 1,
        )
        rate = legacy_square_rate(Task(0, 0, "square", q, "shallow", ""))
    else:
        blocks = 2 if architecture == "res16" else 4
        width = min(
            range(1, 256), key=lambda w: (abs(residual_count(w, blocks) - target), w)
        )
        p = residual_count(width, blocks)
        result.update(
            width=width,
            blocks=blocks,
            architecture_params=p,
            trainable_params=p,
            readout_params=4 * width + 1,
        )
        rate = legacy_square_rate(Task(0, 0, "square", width, "residual", ""))
    result.update(
        actual_kappa=result["trainable_params"] / n,
        learning_rate=None if architecture == "rf_readout" else rate,
    )
    return result


def legacy_config(task):
    a = task["architecture"]
    return ArchitectureConfig(
        q=task["q"],
        architecture={"cnn2": "two_layer", "cnn4": "shallow", "res16": "residual"}[a],
        target_params=task["architecture_params"],
        c1=task.get("c1", 0),
        c2=task.get("c2", 0),
        shallow_channels=tuple(task.get("channels", (0, 0, 0, 0))),
        residual_base=task.get("width", 0),
    )


def build_legacy(task, tf):
    (seed, a) = (task["model_seed"], task["architecture"])
    if a in ("cnn2", "cnn4", "res16"):
        model = build_model(legacy_config(task), seed)
    else:
        inputs = tf.keras.Input((32, 32, 3), name="image")
        if a == "cnn6":
            x = inputs
            for i, (c, stride) in enumerate(
                zip(task["channels"], (1, 2, 2, 1, 1, 1)), 1
            ):
                x = legacy_block(x, c, stride, seed + i, f"block{i}")
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
            x = tf.keras.layers.LayerNormalization(name="feature_layer_norm")(x)
            head_seed = seed + 7
        elif a == "res28":
            x = seeded_conv(16, 3, 1, seed + 1, "stem_conv")(inputs)
            layer_seed = seed + 10
            for group, c in enumerate(
                (task["width"], 2 * task["width"], 4 * task["width"])
            ):
                for block in range(4):
                    stride = 2 if group > 0 and block == 0 else 1
                    x = wrn_block(
                        x, c, stride, layer_seed, f"g{group + 1}_b{block + 1}"
                    )
                    layer_seed += 10
            x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
            x = tf.keras.layers.ReLU(name="head_relu")(x)
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
            head_seed = seed + 200
        else:
            raise ValueError(a)
        score = tf.keras.layers.Dense(
            1,
            use_bias=True,
            name="score",
            kernel_initializer=tf.keras.initializers.HeNormal(seed=head_seed),
        )(x)
        model = tf.keras.Model(inputs, score, name=a)
    actual = sum((int(np.prod(v.shape)) for v in model.trainable_variables))
    if actual != task["trainable_params"]:
        raise RuntimeError(
            f"Parameter count mismatch: {actual} != {task['trainable_params']}"
        )
    return model


def configure_runtime():
    if tf.__version__ != "2.18.0":
        raise RuntimeError(
            "Reference TensorFlow is 2.18.0; use it for faithful reproduction"
        )
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if N_TOTAL == 7000:
        tf.config.experimental.enable_op_determinism()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def load_data(cifar_dir=None):
    
    if cifar_dir is not None:
        os.environ["CIFAR10_DIR"] = str(cifar_dir)
    split = make_raf_split(
        DataConfig(alpha=N_TOTAL / 3072, data_seed=0, eps=0.2, n_test=2000, n_val=1000)
    )
    if (
        N_TOTAL == 6144
        and split["index_digest"]
        != "e94827f588241d144c6836ddc65b883ec4daa1a494745e733708ac0453d3365b"
    ):
        raise RuntimeError("The original BCE dataset has not been reproduced")
    return split


def scores(model, array, batch=256):
    return np.concatenate(
        [
            np.asarray(model(array[i : i + batch], training=False)).reshape(-1)
            for i in range(0, len(array), batch)
        ]
    )


def classification_metrics(train, test, split):
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise FloatingPointError(
            "Nonfinite scores; do not silently discard a trajectory"
        )
    y, yt, mask = split["y_train_pm1"], split["y_test_pm1"], split["mem_mask"]
    pred, pred_test = np.where(train >= 0, 1.0, -1.0), np.where(test >= 0, 1.0, -1.0)
    return dict(
        E_gen=float(np.mean(pred_test != yt)),
        E_mem=float(np.mean(pred[mask] != y[mask])),
        E_train=float(np.mean(pred != y)),
        E_rule_train=float(np.mean(pred[~mask] != y[~mask])),
    )


def arithmetic_coordinates(values, window=DISPLAY_WINDOW):
    """Average up to `window` saved observations per seed with equal weights.
    At index i use radius min(window//2, i, n_times-1-i); endpoints stay raw.
    Average seeds afterwards; do not time-smooth the appendix trajectories."""
    if not isinstance(window, int) or window < 1 or window % 2 != 1:
        raise ValueError("Window must be a positive odd integer")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite display coordinates")
    result = values.copy()
    for i in range(len(values)):
        r = min(window // 2, i, len(values) - 1 - i)
        if r:
            result[i] = sum(values[i + j] for j in range(-r, r + 1)) / (2 * r + 1)
    return result


def plot_trajectories(frames, prefix, include_test_loss=False):
    
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    raw = pd.concat(frames, ignore_index=True)
    if raw.duplicated(["architecture", "kappa", "seed", "time"]).any():
        raise ValueError("Duplicate seed/time observations")
    curves = {}
    for (k, arch), group in raw.groupby(["kappa", "architecture"]):
        runs = [g.sort_values("time") for _, g in group.groupby("seed")]
        times = runs[0].time.to_numpy()
        if any(not np.array_equal(g.time.to_numpy(), times) for g in runs):
            raise ValueError("Use a common recorded horizon/grid within each curve")
        values = np.stack([g[list(METRICS)].to_numpy() for g in runs])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite observations")
        half = (
            student_t.ppf(0.975, len(runs) - 1)
            * values.std(axis=0, ddof=1)
            / np.sqrt(len(runs))
            if len(runs) > 1
            else np.zeros_like(values[0])
        )
        display = np.stack([arithmetic_coordinates(v[:, :2]) for v in values]).mean(
            axis=0
        )
        curves[k, arch] = (times, values.mean(axis=0), half, display)
    capacities = sorted(raw.kappa.unique())
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(capacities), figsize=(6 * len(capacities), 4.8), squeeze=False
    )
    for k, ax in zip(capacities, axes[0]):
        for arch in MAIN_ARCHS:
            if (k, arch) not in curves:
                continue
            times, mean, half, disp = curves[k, arch]
            best = int(disp[:, 0].argmin())
            color = COLORS[arch]
            ax.plot(disp[:, 0], disp[:, 1], color=color, label=LABELS[arch])
            ax.plot(*disp[0], "^", color=color)
            ax.plot(*disp[-1], "s", color=color, mfc="white", zorder=4)
            ax.plot(*disp[best], "x", color=color, zorder=5)
        ax.set(
            xlabel="Test error $E_{gen}$",
            ylabel="Memorization error $E_{mem}$",
            ylim=(-0.018, 0.52),
        )
        ax.set_yticks(np.arange(0, 0.51, 0.1))
        ax.set_title(r"$\kappa\approx %g$" % k)
        ax.grid(alpha=0.2)
    handles = [
        Line2D([], [], color=COLORS[a], label=LABELS[a])
        for a in MAIN_ARCHS
        if a in set(raw.architecture)
    ]
    if handles:
        fig.legend(handles=handles, loc="upper center", ncol=min(4, len(handles)))
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(str(prefix) + "_parametric.pdf")
    plt.close(fig)
    for family in ("CNN", "RF"):
        arches = [
            a
            for a in ARCHS
            if (a == "rf_readout") == (family == "RF") and a in set(raw.architecture)
        ]
        if not arches:
            continue
        family_capacities = [
            k for k in capacities if any((k, a) in curves for a in arches)
        ]
        metric_names = (
            ("E_gen", "test_loss", "E_mem", "train_loss")
            if include_test_loss
            else ("E_gen", "E_mem", "train_loss")
        )
        titles = dict(
            E_gen="Test error",
            test_loss="Test loss",
            E_mem="Memorization error",
            train_loss="Training loss",
        )
        square_layout = N_TOTAL == 7000
        shape = (
            (len(metric_names), len(family_capacities))
            if square_layout
            else (len(family_capacities), len(metric_names))
        )
        fig, axes = plt.subplots(
            *shape, figsize=(5 * shape[1], 3.5 * shape[0]), squeeze=False
        )
        for row, k in enumerate(family_capacities):
            for arch in arches:
                if (k, arch) not in curves:
                    continue
                times, mean, half, _ = curves[k, arch]
                best = int(mean[:, 0].argmin())
                for col, metric in enumerate(metric_names):
                    j = METRICS.index(metric)
                    ax = axes[col, row] if square_layout else axes[row, col]
                    y = mean[:, j]
                    visible = (
                        np.where(y > 0, y, np.nan) if metric == "train_loss" else y
                    )
                    x = np.log10(1 + times)
                    ax.plot(x, visible, color=COLORS[arch])
                    lower, upper = mean[:, j] - half[:, j], mean[:, j] + half[:, j]
                    if metric == "train_loss":
                        lower = np.maximum(lower, np.min(y[y > 0]) * 0.1)
                    else:
                        lower = np.maximum(lower, 0)
                    ax.fill_between(x, lower, upper, color=COLORS[arch], alpha=0.1)
                    ax.plot(x[-1], visible[-1], "s", color=COLORS[arch], ms=4)
                    ax.plot(x[best], visible[best], "x", color=COLORS[arch], ms=5)
            max_time = max(
                c[0][-1] for (cap, a), c in curves.items() if cap == k and a in arches
            )
            times_ticks = (
                [0, 1000, 1000000, 1000000000, max_time]
                if family == "RF"
                else [0, 1, 10, 100, max_time]
            )
            times_ticks = sorted({t for t in times_ticks if t <= max_time})
            for col, metric in enumerate(metric_names):
                ax = axes[col, row] if square_layout else axes[row, col]
                if square_layout:
                    if col == 0:
                        ax.set_title(r"$\kappa\approx %g$" % k)
                    ax.set_ylabel(titles[metric])
                else:
                    ax.set_title(titles[metric])
                ax.set_xlabel(
                    "Full-batch GD step" if family == "RF" else "Training epoch"
                )
                ax.set_xticks(
                    np.log10(1 + np.array(times_ticks)),
                    [str(int(t)) for t in times_ticks],
                )
                if metric == "train_loss":
                    ax.set_yscale("log")
                else:
                    ax.set_ylim(bottom=-0.018 if metric.startswith("E_") else 0)
                ax.grid(alpha=0.2)
        fig.legend(
            handles=[Line2D([], [], color=COLORS[a], label=LABELS[a]) for a in arches],
            loc="upper center",
            ncol=len(arches),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(str(prefix) + "_time_" + family + ".pdf")
        plt.close(fig)


def feature_blocks(task, data, block_size=512):
    (seed, c1, c2) = (task["model_seed"], task["c1"], task["c2"])
    inputs = tf.keras.Input((32, 32, 3))
    x = tf.keras.layers.AveragePooling2D(8)(inputs)
    x = seeded_conv(c1, 3, 1, seed + 1, "conv3x3")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    stem = tf.keras.Model(inputs, x)
    arrays = []
    for key in ("X_train", "X_test"):
        images = data[key]
        arrays.append(
            np.concatenate(
                [
                    stem(images[i : i + 256], training=False).numpy()
                    for i in range(0, len(images), 256)
                ]
            )
        )
    kernel = (
        tf.keras.initializers.HeNormal(seed=seed + 2)((1, 1, c1, c2))
        .numpy()
        .reshape(c1, c2)
    )
    for start in range(0, c2, block_size):
        weights = tf.convert_to_tensor(kernel[:, start : start + block_size])
        yield tuple((tf.nn.relu(tf.matmul(a, weights)).numpy() for a in arrays))


def spectral(K, Ktest, y, times, rank_bound=None):
    (eigen, U) = eigh(K, check_finite=True, driver="evd")
    tolerance = 100 * np.finfo(float).eps * len(K) * max(1.0, float(eigen[-1]))
    if eigen[0] < -tolerance:
        raise RuntimeError("Gram matrix is not numerically PSD")
    negative = int(np.sum(eigen < 0))
    eigen = np.maximum(eigen, 0)
    rank_bound = len(K) if rank_bound is None else min(len(K), int(rank_bound))
    null_count = len(K) - rank_bound
    if null_count:
        if eigen[null_count - 1] > tolerance:
            raise RuntimeError("Gram rank exceeds feature-count bound")
        eigen[:null_count] = 0.0
    eta = 0.9 / eigen[-1]
    times = np.asarray(times, dtype=float)
    response = -np.expm1(np.log1p(-eta * eigen)[:, None] * times[None, :])
    gain = np.divide(
        response, eigen[:, None], out=np.zeros_like(response), where=eigen[:, None] > 0
    )
    coeff = U.T @ y
    train = U @ (coeff[:, None] * response)
    test = Ktest @ U @ (coeff[:, None] * gain)
    small = (eigen > 0) & (eigen <= tolerance)
    small_train = U[:, small] @ (coeff[small, None] * response[small])
    small_test = Ktest @ U[:, small] @ (coeff[small, None] * gain[small])
    sensitivity = max(
        float(np.max(np.abs(small_train))), float(np.max(np.abs(small_test)))
    )
    if sensitivity > 1e-05:
        raise RuntimeError(
            f"RF path is sensitive to unresolved Gram modes ({sensitivity:g}); do not silently truncate. Use a direct-SVD/high-precision follow-up."
        )
    direct_error = 0.0
    if 1 in times:
        j = int(np.flatnonzero(times == 1)[0])
        direct_error = max(
            float(np.max(np.abs(train[:, j] - eta * K @ y))),
            float(np.max(np.abs(test[:, j] - eta * Ktest @ y))),
        )
        if direct_error > 1e-07:
            raise RuntimeError("Spectral trajectory fails direct first-step check")
    return (
        train,
        test,
        dict(
            learning_rate=float(eta),
            largest_eigenvalue=float(eigen[-1]),
            numerical_rank=int(np.sum(eigen > tolerance)),
            rank_tolerance=float(tolerance),
            positive_modes_retained=int(np.sum(eigen > 0)),
            negative_roundoff_modes_clipped=negative,
            direct_first_step_max_error=direct_error,
            algebraic_rank_bound=rank_bound,
            finite_horizon=int(max(times)),
            unresolved_mode_prediction_sensitivity=sensitivity,
            solver="float64_gram_eigh",
            numerical_pseudoinverse_half_mse=float(
                0.5 * np.mean((U @ (coeff * (eigen > tolerance)) - y) ** 2)
            ),
        ),
    )


def spectral_features(F, Ftest, y, times):
    """Direct SVD for the finite-step full-batch square-loss GD trajectory."""
    n = len(F)
    (U, s, Vt) = svd(F, full_matrices=False, check_finite=True, lapack_driver="gesdd")
    eigen = s * s / n
    eta = 0.9 / eigen[0]
    times = np.asarray(times, dtype=float)
    response = -np.expm1(np.log1p(-eta * eigen)[:, None] * times[None, :])
    coeff = U.T @ y
    gain = np.divide(
        response, s[:, None], out=np.zeros_like(response), where=s[:, None] > 0
    )
    tr = U @ (coeff[:, None] * response)
    te = Ftest @ Vt.T @ (coeff[:, None] * gain)
    tol = np.finfo(float).eps * max(F.shape) * s[0]
    first = 0.0
    if 1 in times:
        j = int(np.flatnonzero(times == 1)[0])
        w = eta * (F.T @ y) / n
        first = max(
            float(np.max(np.abs(tr[:, j] - F @ w))),
            float(np.max(np.abs(te[:, j] - Ftest @ w))),
        )
        if first > 1e-07:
            raise RuntimeError("SVD path failed direct GD check")
    return (
        tr,
        te,
        dict(
            solver="float64_feature_svd",
            learning_rate=float(eta),
            numerical_rank=int(np.sum(s > tol)),
            singular_value_tolerance=float(tol),
            positive_modes_retained=int(np.sum(s > 0)),
            direct_first_step_max_error=first,
            finite_horizon=int(max(times)),
            numerical_pseudoinverse_half_mse=float(
                0.5 * np.mean((U @ (coeff * (s > tol)) - y) ** 2)
            ),
        ),
    )


def configuration(architecture, kappa=100, seed=0):
    if architecture not in ARCHS or kappa not in (1, 10, 100):
        raise ValueError("Choose an available architecture and kappa in (1,10,100)")
    task = legacy_configuration(architecture, kappa, N_TOTAL)
    return dict(task, seed=seed, model_seed=INIT_SEED_BASE + 100000 * seed)


def build(task):
    return build_legacy(task, tf)


def optimizer(task):
    return tf.keras.optimizers.SGD(
        learning_rate=task["learning_rate"], momentum=0.9, nesterov=True, clipnorm=5.0
    )


def metrics(train, test, split):
    train, test = np.asarray(train, dtype=np.float64), np.asarray(
        test, dtype=np.float64
    )
    return dict(
        classification_metrics(train, test, split),
        train_loss=float(np.mean((train - split["y_train_pm1"]) ** 2)),
        test_loss=float(np.mean((test - split["y_test_pm1"]) ** 2)),
    )


def evaluate(model, split):
    return metrics(
        scores(model, split["X_train"]), scores(model, split["X_test"]), split
    )


def make_step(model, opt, task=None):
    @tf.function(reduce_retracing=True)
    def step(x, y):
        with tf.GradientTape() as tape:
            pred = tf.reshape(model(x, training=True), (-1,))
            loss = tf.reduce_mean(tf.square(pred - y))
        grads = tape.gradient(loss, model.trainable_variables)
        tf.debugging.assert_all_finite(loss, "Nonfinite square loss")
        for grad in grads:
            tf.debugging.assert_all_finite(grad, "Nonfinite gradient")
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    return step


def recorded_times(epochs):
    return sorted(
        set(range(min(epochs, 100) + 1))
        | set(range(105, min(epochs, 500) + 1, 5))
        | set(range(525, epochs + 1, 25))
        | {epochs}
    )


def rf_trajectory(task, split, max_steps=10**12):
    
    n, nt, d = len(split["X_train"]), len(split["X_test"]), task["c2"]
    direct = d + 1 <= n
    if direct:
        F, Ft = np.ones((n, d + 1)), np.ones((nt, d + 1))
    else:
        K, Kt = np.ones((n, n)) / n, np.ones((nt, n)) / n
    seen = 0
    for a, b in feature_blocks(task, split):
        mean = a.mean(axis=0, dtype=np.float64).astype("float32")
        std = a.std(axis=0, dtype=np.float64).astype("float32")
        std[std < 1e-6] = 1.0
        scale = np.float32(1 / np.sqrt(d))
        a, b = ((a - mean) / std * scale).astype("float64"), (
            (b - mean) / std * scale
        ).astype("float64")
        if direct:
            F[:, seen : seen + a.shape[1]], Ft[:, seen : seen + a.shape[1]] = a, b
        else:
            K += (a @ a.T) / n
            Kt += (b @ a.T) / n
        seen += a.shape[1]
    if seen != d:
        raise RuntimeError("Incorrect feature dimension")
    del a, b
    gc.collect()
    times = sorted(
        {t for t in (0, 1, 2, 3, 5, 8, 10) if t <= max_steps}
        | (
            {int(round(v)) for v in np.geomspace(12, max_steps, 240)}
            if max_steps >= 12
            else {max_steps}
        )
    )
    y = split["y_train_pm1"].astype(float)
    train, test, diag = (
        spectral_features(F, Ft, y, times)
        if direct
        else spectral(K, Kt, y, times, rank_bound=d + 1)
    )
    eta = (
        diag["learning_rate"] / 2
    )  
    frame = pd.DataFrame(
        [
            dict(
                architecture="rf_readout",
                kappa=task["kappa"],
                actual_kappa=task["actual_kappa"],
                seed=task["seed"],
                time=t,
                updates=t,
                learning_rate=eta,
                **metrics(train[:, i], test[:, i], split),
            )
            for i, t in enumerate(times)
        ]
    )
    if (np.diff(frame.train_loss) > 1e-9 * max(1, frame.train_loss.max())).any():
        raise RuntimeError("Stable spectral GD must not increase training MSE")
    return frame


def run_trajectory(architecture="cnn4", epochs=2, seed=0, kappa=100, split=None):
    
    if epochs < 1 or seed < 0:
        raise ValueError("epochs must be positive and seed nonnegative")
    split = load_data() if split is None else split
    if len(split["X_train"]) != N_TOTAL:
        raise ValueError("Wrong training dataset size")
    task = configuration(architecture, kappa=kappa, seed=seed)
    tf.keras.backend.clear_session()
    set_all_seeds(task["model_seed"])
    if architecture == "rf_readout":
        return rf_trajectory(task, split)
    model = build(task)
    if count_params(model.trainable_variables) != task["trainable_params"]:
        raise RuntimeError(
            "Trainable parameter count differs from the recorded architecture"
        )
    opt = optimizer(task)
    opt.build(model.trainable_variables)
    step = make_step(model, opt, task)
    rows = []

    def record(epoch):
        row = dict(
            architecture=architecture,
            kappa=task["kappa"],
            actual_kappa=task["actual_kappa"],
            seed=seed,
            time=epoch,
            updates=int(opt.iterations.numpy()),
            learning_rate=float(opt.learning_rate.numpy()),
            **evaluate(model, split),
        )
        rows.append(row)
        print(
            f"epoch={epoch} test={row['E_gen']:.5f} mem={row['E_mem']:.5f} "
            f"train_loss={row['train_loss']:.7g}",
            flush=True,
        )

    record(0)
    schedule = set(recorded_times(epochs))
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(task["model_seed"] + 12345 + epoch).permutation(
            N_TOTAL
        )
        for start in range(0, N_TOTAL, 128):
            ids = order[start : start + 128]
            args = (split["X_train"][ids], split["y_train_pm1"][ids])
            if N_TOTAL == 6144:
                step(*args, ids.astype(np.int32), tf.constant(epoch, tf.int32))
            else:
                step(*args)
        if epoch in schedule:
            record(epoch)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-dir", type=Path)
    parser.add_argument("--architecture", choices=ARCHS, default="cnn4")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    if N_TOTAL == 7000:
        parser.add_argument("--kappa", type=int, choices=(1, 10, 100), default=100)
    parser.add_argument("--output", type=Path, default=Path("trajectory.csv"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    data = load_data(args.cifar_dir)
    frame = run_trajectory(
        args.architecture,
        args.epochs,
        args.seed,
        kappa=getattr(args, "kappa", 100),
        split=data,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                data_counts=data["counts"],
                index_digest=data["index_digest"],
                model=configuration(
                    args.architecture, kappa=getattr(args, "kappa", 100), seed=args.seed
                ),
                software=dict(
                    tensorflow=tf.__version__,
                    keras=tf.keras.__version__,
                    numpy=np.__version__,
                ),
                display_window=DISPLAY_WINDOW,
            ),
            indent=2,
        )
        + "\n"
    )
    print("CSV:", args.output)
    if args.plot:
        plot_trajectories([frame], args.output.with_suffix(""))


if __name__ == "__main__":
    main()
