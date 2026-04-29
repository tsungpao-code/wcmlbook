# q7bc_csinet_cost2100.py
# Q7(b)(c): CsiNet NMSE evaluation and mixed-dataset training
# Dataset source: COST2100 datasets generated in Q7(a)

import os
import math
import numpy as np
import scipy.io as sio
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, BatchNormalization, LeakyReLU, Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. Basic Settings
# ============================================================

DATA_DIR = os.path.join(
    "cost2100",
    "cost2100-master",
    "cost2100-master",
    "matlab",
    "q7_generated_datasets"
)

RESULT_DIR = "result"
MODEL_DIR = "saved_model"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# CsiNet input setting
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 2
CSI_VECTOR_LEN = IMG_HEIGHT * IMG_WIDTH  # 1024 complex entries
ENCODED_DIM = 512                        # compression dimension
RESIDUAL_NUM = 2

# Training setting
EPOCHS = 30          # If your computer is slow, change to 5 or 10 first.
BATCH_SIZE = 32
RANDOM_SEED = 2026

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


DATASETS = {
    "D1_indoor_uniform": "D1_indoor_uniform.mat",
    "D2_indoor_center": "D2_indoor_center.mat",
    "D3_indoor_edge": "D3_indoor_edge.mat",
    "D4_indoor_hotspot": "D4_indoor_hotspot.mat",
    "D5_indoor_ring": "D5_indoor_ring.mat",
    "D6_indoor_line": "D6_indoor_line.mat",
}


# ============================================================
# 2. Data Loading and Formatting
# ============================================================

def read_mat_variable(mat_path, var_name):
    try:
        data = sio.loadmat(mat_path)
        if var_name in data:
            return data[var_name]
        return None
    except NotImplementedError:
        with h5py.File(mat_path, "r") as f:
            if var_name not in f:
                return None

            arr = np.array(f[var_name])

            if arr.dtype.fields is not None:
                if "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
                    arr = arr["real"] + 1j * arr["imag"]

            return arr.T


def load_cost2100_dataset(mat_path):
    H = read_mat_variable(mat_path, "H_norm")

    if H is None:
        H = read_mat_variable(mat_path, "H_complex")
        if H is None:
            raise KeyError(f"No H_norm or H_complex found in {mat_path}")

        H = H / (np.max(np.abs(H)) + 1e-12)

    H = np.asarray(H)

    if H.ndim != 2:
        H = H.reshape(H.shape[0], -1)

    if H.shape[0] > H.shape[1]:
        H = H.T

    num_samples = H.shape[0]
    csi_dim = H.shape[1]

    H_fixed = np.zeros((num_samples, CSI_VECTOR_LEN), dtype=np.complex64)

    if csi_dim >= CSI_VECTOR_LEN:
        H_fixed[:, :] = H[:, :CSI_VECTOR_LEN]
    else:
        H_fixed[:, :csi_dim] = H

    H_real = np.real(H_fixed).astype("float32")
    H_imag = np.imag(H_fixed).astype("float32")

    H_real = H_real.reshape(num_samples, IMG_HEIGHT, IMG_WIDTH)
    H_imag = H_imag.reshape(num_samples, IMG_HEIGHT, IMG_WIDTH)

    X = np.stack([H_real, H_imag], axis=-1)
    return X

def split_dataset(X, train_ratio=0.7, val_ratio=0.15):
    """
    Split one dataset into train, validation, and test sets.
    """
    n = X.shape[0]
    idx = np.random.permutation(n)
    X = X[idx]

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train = X[:n_train]
    X_val = X[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]

    return X_train, X_val, X_test


def load_all_datasets():
    """
    Load all six datasets generated in Q7(a).
    """
    all_data = {}

    for name, filename in DATASETS.items():
        mat_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Dataset not found: {mat_path}")

        X = load_cost2100_dataset(mat_path)
        X_train, X_val, X_test = split_dataset(X)

        all_data[name] = {
            "train": X_train,
            "val": X_val,
            "test": X_test,
        }

        print(f"{name}: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    return all_data


# ============================================================
# 3. CsiNet Model
# ============================================================

def residual_block(x):
    """
    Residual block used in the decoder.
    """
    shortcut = x

    y = Conv2D(8, kernel_size=(3, 3), padding="same")(x)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(16, kernel_size=(3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(IMG_CHANNELS, kernel_size=(3, 3), padding="same")(y)
    y = BatchNormalization()(y)

    y = Add()([shortcut, y])
    y = LeakyReLU()(y)

    return y


def build_csinet():
    """
    Build a simplified CsiNet-style autoencoder.

    Encoder:
        Conv2D -> BatchNorm -> LeakyReLU -> Flatten -> Dense

    Decoder:
        Dense -> Reshape -> Residual Blocks -> Conv2D
    """
    input_csi = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Encoder
    x = Conv2D(IMG_CHANNELS, kernel_size=(3, 3), padding="same")(input_csi)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    encoded = Dense(ENCODED_DIM, activation="linear", name="encoded_vector")(x)

    # Decoder
    x = Dense(IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS, activation="linear")(encoded)
    x = Reshape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))(x)

    for _ in range(RESIDUAL_NUM):
        x = residual_block(x)

    output_csi = Conv2D(
        IMG_CHANNELS,
        kernel_size=(3, 3),
        padding="same",
        activation="linear"
    )(x)

    model = Model(inputs=input_csi, outputs=output_csi)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    return model


# ============================================================
# 4. NMSE Evaluation
# ============================================================

def to_complex_matrix(X):
    """
    Convert [N, 32, 32, 2] real/imag representation to complex matrix [N, 1024].
    """
    real = X[..., 0].reshape(X.shape[0], -1)
    imag = X[..., 1].reshape(X.shape[0], -1)
    return real + 1j * imag


def nmse_db(y_true, y_pred):
    """
    Compute NMSE in dB.
    NMSE = ||H - H_hat||^2 / ||H||^2
    """
    H_true = to_complex_matrix(y_true)
    H_pred = to_complex_matrix(y_pred)

    power = np.sum(np.abs(H_true) ** 2, axis=1)
    mse = np.sum(np.abs(H_true - H_pred) ** 2, axis=1)

    nmse = np.mean(mse / (power + 1e-12))
    return 10 * math.log10(nmse + 1e-12)


def evaluate_model_on_all_datasets(model, all_data):
    """
    Evaluate one trained CsiNet model on D1~D6.
    """
    results = {}

    for dataset_name, data in all_data.items():
        X_test = data["test"]
        X_hat = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        score = nmse_db(X_test, X_hat)
        results[dataset_name] = score
        print(f"{dataset_name}: NMSE = {score:.4f} dB")

    return results


# ============================================================
# 5. Training Functions for Q7(b) and Q7(c)
# ============================================================

def train_csinet(X_train, X_val, model_name):
    """
    Train CsiNet model.
    """
    model = build_csinet()

    save_path = os.path.join(MODEL_DIR, model_name + ".keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    # Save loss curves
    train_loss_path = os.path.join(RESULT_DIR, model_name + "_train_loss.csv")
    val_loss_path = os.path.join(RESULT_DIR, model_name + "_val_loss.csv")

    np.savetxt(train_loss_path, np.array(history.history["loss"]), delimiter=",")
    np.savetxt(val_loss_path, np.array(history.history["val_loss"]), delimiter=",")

    return model


def run_q7b_single_dataset_training(all_data):
    """
    Q7(b):
    Train CsiNet on D1 only, then test on D1~D6.
    """
    print("\n==============================")
    print("Q7(b): Single-dataset training")
    print("Training dataset: D1_indoor_uniform")
    print("==============================")

    X_train = all_data["D1_indoor_uniform"]["train"]
    X_val = all_data["D1_indoor_uniform"]["val"]

    model = train_csinet(
        X_train,
        X_val,
        model_name="q7b_single_D1_csinet"
    )

    print("\nQ7(b) evaluation on all datasets:")
    results = evaluate_model_on_all_datasets(model, all_data)

    return results


def run_q7c_mixed_dataset_training(all_data):
    """
    Q7(c):
    Mix D1~D6 as training data, then test on D1~D6.
    """
    print("\n==============================")
    print("Q7(c): Mixed-dataset training")
    print("Training dataset: D1 + D2 + D3 + D4 + D5 + D6")
    print("==============================")

    X_train_mix = np.concatenate(
        [data["train"] for data in all_data.values()],
        axis=0
    )

    X_val_mix = np.concatenate(
        [data["val"] for data in all_data.values()],
        axis=0
    )

    # Shuffle mixed training set
    idx_train = np.random.permutation(X_train_mix.shape[0])
    idx_val = np.random.permutation(X_val_mix.shape[0])

    X_train_mix = X_train_mix[idx_train]
    X_val_mix = X_val_mix[idx_val]

    print(f"Mixed train shape: {X_train_mix.shape}")
    print(f"Mixed val shape: {X_val_mix.shape}")

    model = train_csinet(
        X_train_mix,
        X_val_mix,
        model_name="q7c_mixed_csinet"
    )

    print("\nQ7(c) evaluation on all datasets:")
    results = evaluate_model_on_all_datasets(model, all_data)

    return results


# ============================================================
# 6. Main
# ============================================================

def main():
    print("Loading Q7(a) COST2100 datasets...")
    all_data = load_all_datasets()

    # Q7(b)
    results_b = run_q7b_single_dataset_training(all_data)

    # Q7(c)
    results_c = run_q7c_mixed_dataset_training(all_data)

    # Save comparison results
    rows = []

    for dataset_name in DATASETS.keys():
        b_nmse = results_b[dataset_name]
        c_nmse = results_c[dataset_name]
        improvement = b_nmse - c_nmse  # positive means C is better because NMSE dB is lower

        rows.append({
            "Testing Dataset": dataset_name,
            "Q7b Single-Dataset Training NMSE (dB)": b_nmse,
            "Q7c Mixed-Dataset Training NMSE (dB)": c_nmse,
            "Improvement (B - C, dB)": improvement
        })

    df = pd.DataFrame(rows)

    output_csv = os.path.join(RESULT_DIR, "q7bc_nmse_results.csv")
    df.to_csv(output_csv, index=False)

    print("\n==============================")
    print("Final Q7(b)(c) NMSE Results")
    print("==============================")
    print(df)
    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    main()