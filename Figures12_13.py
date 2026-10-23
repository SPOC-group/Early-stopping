"""ResNet controls for Figures 12 and 13."""
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

@dataclass(frozen=True)
class Config:
    data_seed: int = 0
    init_seed_base: int = 98765
    n_init_seeds: int = 5
    alpha: float = 2.0
    eps: float = 0.2
    n_test: int = 2000
    class_pos: int = 3
    class_neg: int = 5
    class_mem: int = 1
    preprocess_mode: str = 'zscore'
    activations: tuple = ('relu', 'swish')
    wrn_depth: int = 16
    widen_factor: int = 4
    dropout_rate: float = 0.0
    weight_decay: float = 0.0
    epochs: int = 300
    batch_size: int = 128
    eval_every: int = 5
    optimizer: str = 'sgd_nesterov'
    learning_rate: float = 0.05
    momentum: float = 0.9
    warmup_epochs: int = 5
    cosine_alpha: float = 0.01
    clipnorm: float = 5.0
    use_augmentation: bool = True
    crop_padding: int = 4
    output_dir: str = 'CNN_new'

CFG = Config()

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def find_cifar10_dir() -> Path:
    candidates = []
    if os.environ.get('CIFAR10_DIR'):
        candidates.append(Path(os.environ['CIFAR10_DIR']))
    keras_home = Path(os.environ.get('KERAS_HOME', '.keras'))
    candidates.append(keras_home / 'datasets' / 'cifar-10-batches-py-target' / 'cifar-10-batches-py')
    candidates.append(Path.cwd() / '.keras' / 'datasets' / 'cifar-10-batches-py-target' / 'cifar-10-batches-py')
    candidates.append(Path.home() / '.keras' / 'datasets' / 'cifar-10-batches-py-target' / 'cifar-10-batches-py')
    required = [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch', 'batches.meta']
    for base in candidates:
        if all(((base / name).exists() for name in required)):
            return base
    raise FileNotFoundError('Could not find extracted CIFAR10 files. Set CIFAR10_DIR to the folder containing data_batch_1,...,data_batch_5,test_batch,batches.meta.')


def load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with open(path, 'rb') as handle:
            obj = pickle.load(handle, encoding='latin1')
    except Exception as exc:
        raise RuntimeError(f'Could not unpickle CIFAR batch {path}. The CIFAR cache is corrupted; recopy it with rsync --delete.') from exc
    data = obj['data'].astype(np.float32)
    labels = np.asarray(obj['labels'], dtype=np.int64)
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return (images, labels)


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = find_cifar10_dir()
    train_images = []
    train_labels = []
    for i in range(1, 6):
        Xb, yb = load_cifar_batch(base / f'data_batch_{i}')
        train_images.append(Xb)
        train_labels.append(yb)
    X_train_full = np.concatenate(train_images, axis=0).astype(np.float32)
    y_train_full = np.concatenate(train_labels, axis=0).astype(np.int64)
    X_test_full, y_test_full = load_cifar_batch(base / 'test_batch')
    X_test_full = X_test_full.astype(np.float32)
    y_test_full = y_test_full.astype(np.int64)
    if X_train_full.shape != (50000, 32, 32, 3):
        raise RuntimeError(f'Unexpected train shape: {X_train_full.shape}')
    if X_test_full.shape != (10000, 32, 32, 3):
        raise RuntimeError(f'Unexpected test shape: {X_test_full.shape}')
    return (X_train_full, y_train_full, X_test_full, y_test_full)


def flatten_images(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1).astype(np.float32)


def sample_without_replacement(indices: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices)
    if len(indices) < n:
        raise ValueError(f'Requested {n}, but only {len(indices)} available.')
    return rng.choice(indices, size=n, replace=False)


def fit_preprocessor(X_train_raw: np.ndarray, mode: str='zscore') -> dict:
    X_train_raw = np.asarray(X_train_raw, dtype=np.float32)
    if mode == 'none':
        return {'mode': 'none'}
    if mode == 'zscore':
        mean = X_train_raw.mean(axis=0, keepdims=True).astype(np.float32)
        std = X_train_raw.std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-08] = 1.0
        return {'mode': 'zscore', 'mean': mean, 'std': std}
    raise ValueError(f'Unknown preprocessing mode: {mode}')


def apply_preprocessor(X: np.ndarray, prep: dict) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if prep['mode'] == 'none':
        return X
    if prep['mode'] == 'zscore':
        return (X - prep['mean']) / prep['std']
    raise ValueError(f"Unknown preprocessing mode: {prep['mode']}")


def cifar10_raf_counts(alpha: float, eps: float, d: int, n_test: int) -> dict:
    n_total = int(round(alpha * d))
    n_struct_target = (1.0 - eps) * n_total
    n_struct = int(2 * round(n_struct_target / 2.0))
    n_struct = min(max(n_struct, 2), n_total - 1)
    if n_struct % 2 != 0:
        n_struct += 1 if n_struct < n_total else -1
    n_mem = n_total - n_struct
    if n_test % 2 != 0:
        raise ValueError('n_test must be even.')
    return {'d': int(d), 'n_total': int(n_total), 'n_struct': int(n_struct), 'n_mem': int(n_mem), 'n_struct_pos': int(n_struct // 2), 'n_struct_neg': int(n_struct // 2), 'n_test_pos': int(n_test // 2), 'n_test_neg': int(n_test // 2), 'realized_alpha': float(n_total / d), 'realized_eps': float(n_mem / n_total)}


def activation_layer(x, activation: str, name: str):
    if activation == 'relu':
        return layers.Activation('relu', name=name)(x)
    if activation == 'swish':
        return layers.Activation(tf.keras.activations.swish, name=name)(x)
    raise ValueError(f'Unknown activation: {activation}')


def conv2d(filters: int, kernel_size: int, stride: int, name: str):
    return layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=None, name=name)


def wrn_block(x, filters: int, stride: int, activation: str, dropout_rate: float, name: str):
    shortcut = x
    y = layers.BatchNormalization(name=f'{name}_bn1')(x)
    y = activation_layer(y, activation, name=f'{name}_{activation}_1')
    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = conv2d(filters, 1, stride, name=f'{name}_shortcut')(y)
    y = conv2d(filters, 3, stride, name=f'{name}_conv1')(y)
    y = layers.BatchNormalization(name=f'{name}_bn2')(y)
    y = activation_layer(y, activation, name=f'{name}_{activation}_2')
    if dropout_rate > 0:
        y = layers.Dropout(dropout_rate, name=f'{name}_dropout')(y)
    y = conv2d(filters, 3, 1, name=f'{name}_conv2')(y)
    return layers.Add(name=f'{name}_add')([shortcut, y])


def build_wrn_binary(input_shape, cfg: Config, activation: str) -> tf.keras.Model:
    if (cfg.wrn_depth - 4) % 6 != 0:
        raise ValueError('WRN depth must be 6N+4.')
    n_blocks = (cfg.wrn_depth - 4) // 6
    k = int(cfg.widen_factor)
    inputs = layers.Input(shape=input_shape, name='image')
    x = conv2d(16, 3, 1, name='stem_conv')(inputs)
    for group_idx, filters in enumerate([16 * k, 32 * k, 64 * k]):
        for block_idx in range(n_blocks):
            stride = 1 if group_idx == 0 or block_idx > 0 else 2
            x = wrn_block(x, filters=filters, stride=stride, activation=activation, dropout_rate=cfg.dropout_rate, name=f'g{group_idx + 1}_b{block_idx + 1}')
    x = layers.BatchNormalization(name='head_bn')(x)
    x = activation_layer(x, activation, name=f'head_{activation}')
    x = layers.GlobalAveragePooling2D(name='global_average_pool')(x)
    outputs = layers.Dense(1, activation=None, kernel_initializer='he_normal', kernel_regularizer=None, dtype='float32', name='logit')(x)
    return tf.keras.Model(inputs, outputs, name=f'WRN{cfg.wrn_depth}_{k}x_{activation}_time')


def trainable_params(model: tf.keras.Model) -> int:
    return int(sum((np.prod(v.shape) for v in model.trainable_variables)))


def total_params(model: tf.keras.Model) -> int:
    return int(sum((np.prod(v.shape) for v in model.weights)))


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_lr: float, total_steps: int, warmup_steps: int, alpha: float):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.alpha = float(alpha)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        initial_lr = tf.constant(self.initial_lr, dtype=tf.float32)
        warmup_steps = tf.cast(max(1, self.warmup_steps), tf.float32)
        total_steps = tf.cast(max(self.total_steps, self.warmup_steps + 1), tf.float32)
        warmup_lr = initial_lr * (step + 1.0) / warmup_steps
        progress = (step - warmup_steps) / tf.maximum(1.0, total_steps - warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine = self.alpha + 0.5 * (1.0 - self.alpha) * (1.0 + tf.cos(math.pi * progress))
        cosine_lr = initial_lr * cosine
        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {'initial_lr': self.initial_lr, 'total_steps': self.total_steps, 'warmup_steps': self.warmup_steps, 'alpha': self.alpha}


def make_optimizer(cfg: Config, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    schedule = WarmupCosine(initial_lr=cfg.learning_rate, total_steps=total_steps, warmup_steps=warmup_steps, alpha=cfg.cosine_alpha)
    return tf.keras.optimizers.SGD(learning_rate=schedule, momentum=cfg.momentum, nesterov=True, clipnorm=cfg.clipnorm)


def augment_image(x, y, cfg: Config):
    if cfg.crop_padding > 0:
        pad = int(cfg.crop_padding)
        x = tf.pad(x, [[pad, pad], [pad, pad], [0, 0]], mode='CONSTANT', constant_values=0.0)
        x = tf.image.random_crop(x, size=[32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    return (x, y)


def make_train_dataset(X, y_bin, cfg: Config, seed: int):
    ds = tf.data.Dataset.from_tensor_slices((X, y_bin[:, None]))
    ds = ds.shuffle(len(y_bin), seed=seed, reshuffle_each_iteration=True)
    if cfg.use_augmentation:
        ds = ds.map(lambda x, y: augment_image(x, y, cfg), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return ds.with_options(options)


def sign_predictions_from_logits(logits: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(logits).reshape(-1) >= 0.0, 1.0, -1.0).astype(np.float32)


def classification_error(y_true_pm1, y_pred_pm1) -> float:
    return float(np.mean(np.asarray(y_true_pm1).reshape(-1) != np.asarray(y_pred_pm1).reshape(-1)))


def binary_cross_entropy_from_logits(y_bin: np.ndarray, logits: np.ndarray) -> float:
    y = tf.convert_to_tensor(y_bin.reshape(-1, 1), dtype=tf.float32)
    z = tf.convert_to_tensor(logits.reshape(-1, 1), dtype=tf.float32)
    return float(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z)).numpy())


def atomic_write_csv(df, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


import argparse
import math
import os
import time
import tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

VAL_PER_RULE_CLASS = 500
BASE_BATCH_SIZE = 128
BASE_LEARNING_RATE = .05
ACTIVATION = 'relu'
EPOCHS = 400
EVAL_EVERY = 5
EVAL_BATCH_SIZE = 256
OUT = Path('runs')

def make_conditions() -> list[dict]:
    conditions = [{'condition_key': 'baseline', 'sweep_members': 'n_train,batch,epsilon', 'alpha': 2.0, 'eps': 0.2, 'batch_size': 128, 'lr_policy': 'fixed', 'initial_learning_rate': 0.05}, {'condition_key': 'n_alpha0p5', 'sweep_members': 'n_train', 'alpha': 0.5, 'eps': 0.2, 'batch_size': 128, 'lr_policy': 'fixed', 'initial_learning_rate': 0.05}, {'condition_key': 'n_alpha1', 'sweep_members': 'n_train', 'alpha': 1.0, 'eps': 0.2, 'batch_size': 128, 'lr_policy': 'fixed', 'initial_learning_rate': 0.05}, {'condition_key': 'n_alpha3', 'sweep_members': 'n_train', 'alpha': 3.0, 'eps': 0.2, 'batch_size': 128, 'lr_policy': 'fixed', 'initial_learning_rate': 0.05}, {'condition_key': 'batch512_fixed', 'sweep_members': 'batch', 'alpha': 2.0, 'eps': 0.2, 'batch_size': 512, 'lr_policy': 'fixed', 'initial_learning_rate': 0.05}, {'condition_key': 'batch512_linear', 'sweep_members': 'batch', 'alpha': 2.0, 'eps': 0.2, 'batch_size': 512, 'lr_policy': 'linear', 'initial_learning_rate': 0.2}, {'condition_key': 'batch1024_fixed', 'sweep_members': 'batch', 'alpha': 2.0, 'eps': 0.2, 'batch_size': 1024, 'lr_policy': 'fixed', 'initial_learning_rate': 0.05}, {'condition_key': 'batch1024_linear', 'sweep_members': 'batch', 'alpha': 2.0, 'eps': 0.2, 'batch_size': 1024, 'lr_policy': 'linear', 'initial_learning_rate': 0.4}, {'condition_key': 'eps0p4', 'sweep_members': 'epsilon', 'alpha': 2.0, 'eps': 0.4, 'batch_size': 128, 'lr_policy': 'fixed', 'initial_learning_rate': 0.05}, {'condition_key': 'eps0p6', 'sweep_members': 'epsilon', 'alpha': 2.0, 'eps': 0.6, 'batch_size': 128, 'lr_policy': 'fixed', 'initial_learning_rate': 0.05}, {'condition_key': 'eps0p8', 'sweep_members': 'epsilon', 'alpha': 2.0, 'eps': 0.8, 'batch_size': 128, 'lr_policy': 'fixed', 'initial_learning_rate': 0.05}]
    keys = [condition['condition_key'] for condition in conditions]
    if len(keys) != len(set(keys)):
        raise RuntimeError('Condition keys must be unique.')
    for condition in conditions:
        if condition['lr_policy'] == 'linear':
            expected_lr = BASE_LEARNING_RATE * (condition['batch_size'] / BASE_BATCH_SIZE)
            if not np.isclose(condition['initial_learning_rate'], expected_lr):
                raise RuntimeError(f"Bad linear-scaled LR in {condition['condition_key']}")
    return conditions


def fixed_class_partitions(y_train_full: np.ndarray, y_test_full: np.ndarray, data_seed: int) -> dict:

    def permuted_class_indices(labels, class_id, seed):
        indices = np.where(labels == class_id)[0]
        return np.random.default_rng(seed).permutation(indices)
    train_pos = permuted_class_indices(y_train_full, CFG.class_pos, data_seed + 101)
    train_neg = permuted_class_indices(y_train_full, CFG.class_neg, data_seed + 202)
    train_mem = permuted_class_indices(y_train_full, CFG.class_mem, data_seed + 303)
    val_pos = train_pos[:VAL_PER_RULE_CLASS]
    val_neg = train_neg[:VAL_PER_RULE_CLASS]
    pool_pos = train_pos[VAL_PER_RULE_CLASS:]
    pool_neg = train_neg[VAL_PER_RULE_CLASS:]
    test_pos = np.where(y_test_full == CFG.class_pos)[0]
    test_neg = np.where(y_test_full == CFG.class_neg)[0]
    mem_labels = np.random.default_rng(data_seed + 404).choice([-1.0, 1.0], size=len(train_mem), replace=True).astype(np.float32)
    return {'pool_pos': pool_pos, 'pool_neg': pool_neg, 'pool_mem': train_mem, 'pool_mem_labels': mem_labels, 'val_pos': val_pos, 'val_neg': val_neg, 'test_pos': test_pos, 'test_neg': test_neg}


def make_nested_raf_split(X_train_full: np.ndarray, y_train_full: np.ndarray, X_test_full: np.ndarray, y_test_full: np.ndarray, cfg) -> dict:
    d = int(np.prod(X_train_full.shape[1:]))
    counts = cifar10_raf_counts(cfg.alpha, cfg.eps, d, cfg.n_test)
    partitions = fixed_class_partitions(y_train_full, y_test_full, cfg.data_seed)
    required_counts = {'pool_pos': counts['n_struct_pos'], 'pool_neg': counts['n_struct_neg'], 'pool_mem': counts['n_mem']}
    for pool_name, required in required_counts.items():
        available = len(partitions[pool_name])
        if required > available:
            raise ValueError(f'Condition alpha={cfg.alpha}, eps={cfg.eps} requires {required} samples from {pool_name}, but only {available} remain after reserving the clean validation set.')
    sel_pos = partitions['pool_pos'][:counts['n_struct_pos']]
    sel_neg = partitions['pool_neg'][:counts['n_struct_neg']]
    sel_mem = partitions['pool_mem'][:counts['n_mem']]
    mem_labels = partitions['pool_mem_labels'][:counts['n_mem']]
    X_train_raw = np.concatenate([X_train_full[sel_pos], X_train_full[sel_neg], X_train_full[sel_mem]], axis=0).astype(np.float32)
    y_train_pm1 = np.concatenate([np.ones(counts['n_struct_pos'], dtype=np.float32), -np.ones(counts['n_struct_neg'], dtype=np.float32), mem_labels])
    mem_mask = np.zeros(counts['n_total'], dtype=bool)
    mem_mask[counts['n_struct']:] = True
    X_val_raw = np.concatenate([X_train_full[partitions['val_pos']], X_train_full[partitions['val_neg']]], axis=0).astype(np.float32)
    y_val_pm1 = np.concatenate([np.ones(VAL_PER_RULE_CLASS, dtype=np.float32), -np.ones(VAL_PER_RULE_CLASS, dtype=np.float32)])
    X_test_raw = np.concatenate([X_test_full[partitions['test_pos']], X_test_full[partitions['test_neg']]], axis=0).astype(np.float32)
    y_test_pm1 = np.concatenate([np.ones(len(partitions['test_pos']), dtype=np.float32), -np.ones(len(partitions['test_neg']), dtype=np.float32)])
    prep = fit_preprocessor(flatten_images(X_train_raw), mode=cfg.preprocess_mode)

    def transform(images):
        return apply_preprocessor(flatten_images(images), prep).reshape(-1, 32, 32, 3).astype(np.float32)
    split = {'X_train': transform(X_train_raw), 'y_train_pm1': y_train_pm1.astype(np.float32), 'y_train_bin': ((y_train_pm1 + 1.0) / 2.0).astype(np.float32), 'X_val': transform(X_val_raw), 'y_val_pm1': y_val_pm1.astype(np.float32), 'y_val_bin': ((y_val_pm1 + 1.0) / 2.0).astype(np.float32), 'X_test': transform(X_test_raw), 'y_test_pm1': y_test_pm1.astype(np.float32), 'y_test_bin': ((y_test_pm1 + 1.0) / 2.0).astype(np.float32), 'mem_mask': mem_mask, 'counts': {**counts, 'n_val': int(len(y_val_pm1)), 'n_test': int(len(y_test_pm1))}, '_selected_indices': {'pos': sel_pos.copy(), 'neg': sel_neg.copy(), 'mem': sel_mem.copy()}}
    finite_arrays = [split['X_train'], split['X_val'], split['X_test']]
    if not all((np.isfinite(array).all() for array in finite_arrays)):
        raise FloatingPointError('Non-finite values after preprocessing.')
    return split


def config_for_task(task: dict):
    return replace(CFG, alpha=float(task['alpha']), eps=float(task['eps']), activations=(ACTIVATION,), epochs=EPOCHS, eval_every=EVAL_EVERY, batch_size=int(task['batch_size']), learning_rate=float(task['initial_learning_rate']), output_dir=str(OUT))
def evaluate(model,split):
    predictions=[]
    for key in ('X_train','X_test'):
        predictions.append(np.concatenate([model(split[key][i:i+256],training=False).numpy().reshape(-1)
            for i in range(0,len(split[key]),256)]).astype('float64'))
    train,test=predictions
    if not all(np.isfinite(p).all() for p in predictions):
        raise RuntimeError('Nonfinite inference predictions')
    bad=np.where(train>=0,1.,-1.)!=split['y_train_pm1']
    mem=split['mem_mask']
    return dict(E_gen=float(np.mean(np.where(test>=0,1.,-1.)!=split['y_test_pm1'])),
        E_mem=float(bad[mem].mean()) if mem.any() else None,
        E_train=float(bad.mean()),E_rule_train=float(bad[~mem].mean()),
        train_loss=float(np.mean((train-split['y_train_pm1'])**2)),
        test_loss=float(np.mean((test-split['y_test_pm1'])**2)))


def make_step(tf,model,opt):
    @tf.function(reduce_retracing=True)
    def step(x,y):
        with tf.GradientTape() as tape:
            loss=tf.reduce_mean(tf.square(tf.reshape(model(x,training=True),(-1,))-y))
        gradients=tape.gradient(loss,model.trainable_variables)
        tf.debugging.assert_all_finite(loss,'Nonfinite square loss')
        for g in gradients:
            tf.debugging.assert_all_finite(g,'Nonfinite gradient')
        opt.apply_gradients(zip(gradients,model.trainable_variables))
    return step
def grid(steps, horizon):
    updates = set(range(0, min(horizon, 10)*steps+1))
    updates.update(e*steps for e in range(11, min(horizon, 100)+1))
    updates.update(e*steps for e in range(105, horizon+1, 5))
    updates.add(horizon*steps)
    return sorted(updates)


def conditions(recipe='square'):
    values = make_conditions()
    values += [dict(values[0], condition_key=key, sweep_members='epsilon', eps=eps)
               for key, eps in [('eps0', 0.), ('eps0p1', .1)]]
    if recipe == 'square':
        for task in values:
            task['initial_learning_rate'] = .01 * (task['batch_size']/128 if task['lr_policy'] == 'linear' else 1)
    elif recipe != 'bce':
        raise ValueError('Choose square or bce')
    return values


def run_square(condition='baseline', seed=0, epochs=400, dense=True, split=None,
               output=None):
    if seed < 0 or epochs < 1:
        raise ValueError('Require a nonnegative seed and positive horizon')
    task = next(t for t in conditions('square') if t['condition_key'] == condition)
    cfg = config_for_task(task)
    split = make_nested_raf_split(*load_cifar10(), cfg) if split is None else split
    model_seed = cfg.init_seed_base + 100000*seed
    set_all_seeds(model_seed)
    tf.keras.backend.clear_session()
    model = build_wrn_binary((32, 32, 3), cfg, 'relu')
    if trainable_params(model) != 2746577:
        raise RuntimeError('Wrong ResNet parameter count')
    opt = tf.keras.optimizers.SGD(task['initial_learning_rate'], momentum=.9,
                                      nesterov=True, clipnorm=5.)
    opt.build(model.trainable_variables)
    step = make_step(tf, model, opt)
    steps = math.ceil(len(split['X_train'])/cfg.batch_size)
    wanted = set(grid(steps, epochs) if dense else
                 [e*steps for e in sorted(set(range(0, epochs+1, 5)) | {epochs})])
    rows = []

    def record():
        update = int(opt.iterations.numpy())
        rows.append(dict(condition_key=condition, seed=seed, epoch=update/steps,
                         updates=update, learning_rate=float(opt.learning_rate.numpy()),
                         **evaluate(model, split)))

    record()
    for epoch in range(1, epochs+1):
        order = np.random.default_rng(model_seed+12345+epoch).permutation(len(split['X_train']))
        for start in range(0, len(order), cfg.batch_size):
            ids = order[start:start+cfg.batch_size]
            step(split['X_train'][ids], split['y_train_pm1'][ids])
            if int(opt.iterations.numpy()) in wanted:
                record()
        if output is not None and (epoch % 100 == 0 or epoch == epochs):
            atomic_write_csv(pd.DataFrame(rows), output)
    return pd.DataFrame(rows)


def run_bce(condition='baseline', seed=0, epochs=400, split=None, output=None):
    if seed < 0 or not 1 <= epochs <= 400:
        raise ValueError('BCE controls use a fixed 400-epoch schedule; choose 1..400')
    task = next(t for t in conditions('bce') if t['condition_key'] == condition)
    cfg = config_for_task(task)
    split = make_nested_raf_split(*load_cifar10(), cfg) if split is None else split
    model_seed = cfg.init_seed_base+100000*seed
    set_all_seeds(model_seed)
    tf.keras.backend.clear_session()
    model = build_wrn_binary((32, 32, 3), cfg, 'relu')
    if trainable_params(model) != 2746577:
        raise RuntimeError('Wrong ResNet parameter count')
    model.compile(optimizer=make_optimizer(cfg, math.ceil(len(split['X_train'])/cfg.batch_size)),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    dataset = make_train_dataset(split['X_train'], split['y_train_bin'], cfg,
                                     seed=model_seed+12345)
    class Callback(ControlMetricsCallback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch+1) % 5 == 0 or epoch+1 == epochs:
                self.evaluate_and_checkpoint(epoch+1, logs or {})

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(output) if output is not None else Path(tmp)/'trajectory.csv'
        callback = Callback(split, dict(task, seed=seed), {}, path, cfg)
        model.fit(dataset, epochs=epochs, verbose=0,
                  callbacks=[tf.keras.callbacks.TerminateOnNaN(), callback])
        frame = pd.DataFrame(callback.rows)
    frame['updates'] = frame.optimizer_iterations
    frame['train_loss'], frame['test_loss'] = frame.train_bce, frame.test_bce
    fraction = split['mem_mask'].mean()
    frame['E_train'] = ((fraction*frame.E_mem+(1-fraction)*frame.E_rule_train)
                        if fraction else frame.E_rule_train)
    if not fraction:
        frame['E_mem'] = np.nan
    return frame[['condition_key', 'seed', 'epoch', 'updates', 'learning_rate',
                  'E_gen', 'E_mem', 'E_train', 'E_rule_train', 'train_loss', 'test_loss']]


class ControlMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        split: dict,
        task: dict,
        model_info: dict,
        out_file: Path,
        cfg,
    ):
        super().__init__()
        self.split = split
        self.task = task
        self.model_info = model_info
        self.out_file = Path(out_file)
        self.cfg = cfg
        self.rows = []
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.evaluate_and_checkpoint(epoch=0, logs={})

    def on_epoch_end(self, epoch, logs=None):
        epoch_number = int(epoch + 1)
        if (
            epoch_number % EVAL_EVERY == 0
            or epoch_number == EPOCHS
        ):
            self.evaluate_and_checkpoint(
                epoch=epoch_number,
                logs=logs or {},
            )

    def on_train_end(self, logs=None):
        atomic_write_csv(pd.DataFrame(self.rows), self.out_file)


    def current_lr(self) -> float:
        try:
            learning_rate = self.model.optimizer.learning_rate
            if callable(learning_rate):
                return float(
                    learning_rate(self.model.optimizer.iterations).numpy()
                )
            return float(tf.keras.backend.get_value(learning_rate))
        except Exception:
            return float("nan")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(
            X,
            batch_size=EVAL_BATCH_SIZE,
            verbose=0,
        ).reshape(-1)

    def evaluate_and_checkpoint(self, epoch: int, logs: dict):
        train_logits = self.predict(self.split["X_train"])
        val_logits = self.predict(self.split["X_val"])
        test_logits = self.predict(self.split["X_test"])

        if not all(
            np.isfinite(values).all()
            for values in (train_logits, val_logits, test_logits)
        ):
            raise FloatingPointError(
                "Non-finite predictions during time evaluation."
            )

        train_predictions = sign_predictions_from_logits(train_logits)
        val_predictions = sign_predictions_from_logits(val_logits)
        test_predictions = sign_predictions_from_logits(test_logits)

        mem_mask = self.split["mem_mask"]
        rule_mask = ~mem_mask

        row = {
            **self.task,
            **self.model_info,
            "epoch": int(epoch),
            "optimizer_iterations": int(
                tf.keras.backend.get_value(
                    self.model.optimizer.iterations
                )
            ),
            "learning_rate": float(self.current_lr()),
            "elapsed_sec": float(
                time.time() - self.start_time
                if self.start_time is not None
                else 0.0
            ),
            "fit_loss_logged": float(logs.get("loss", np.nan)),
            "train_bce": binary_cross_entropy_from_logits(
                self.split["y_train_bin"], train_logits
            ),
            "val_bce": binary_cross_entropy_from_logits(
                self.split["y_val_bin"], val_logits
            ),
            "test_bce": binary_cross_entropy_from_logits(
                self.split["y_test_bin"], test_logits
            ),
            "E_gen": classification_error(
                self.split["y_test_pm1"], test_predictions
            ),
            "E_val": classification_error(
                self.split["y_val_pm1"], val_predictions
            ),
            "E_mem": classification_error(
                self.split["y_train_pm1"][mem_mask],
                train_predictions[mem_mask],
            ),
            "E_rule_train": classification_error(
                self.split["y_train_pm1"][rule_mask],
                train_predictions[rule_mask],
            ),
        }
        self.rows.append(row)

        atomic_write_csv(pd.DataFrame(self.rows), self.out_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=['square', 'bce'], default='square')
    parser.add_argument('--condition', default='baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--cifar-dir', required=True)
    parser.add_argument('--output', type=Path, default=Path('trajectory.csv'))
    args = parser.parse_args()
    os.environ['CIFAR10_DIR'] = args.cifar_dir
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options = dict(condition=args.condition, seed=args.seed, epochs=args.epochs, output=args.output)
    frame = (run_square(**options, dense=not args.sparse) if args.recipe == 'square'
             else run_bce(**options))
    atomic_write_csv(frame, args.output)


if __name__ == '__main__':
    main()
