# extra_da_litetcn_csinet.py
# Extra Credit: Doppler-Aware Lightweight Temporal Convolutional CsiNet
# Prototype implementation for replacing CsiNet-LSTM.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Add,
    Flatten,
    DepthwiseConv2D,
    Conv1D,
    Concatenate,
    GlobalAveragePooling1D,
    TimeDistributed,
    Lambda,
)
from tensorflow.keras.models import Model


# ============================================================
# Basic settings
# ============================================================

IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 2
IMG_TOTAL = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS

ENCODED_DIM = 128
SEQ_LEN = 4
RESIDUAL_NUM = 2


# ============================================================
# Residual decoder block
# ============================================================

def residual_decoder_block(x):
    """
    Residual block for CSI reconstruction.
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


# ============================================================
# UE-side lightweight encoder
# ============================================================

def build_lightweight_ue_encoder():
    """
    UE-side lightweight encoder.

    The UE only performs lightweight spatial compression.
    Depthwise separable convolution is used to reduce UE-side computation.
    """
    inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="ue_csi_input")

    x = DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same",
        name="ue_depthwise_conv"
    )(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(
        4,
        kernel_size=(1, 1),
        padding="same",
        name="ue_pointwise_conv"
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    z = Dense(ENCODED_DIM, activation="linear", name="ue_latent_vector")(x)

    return Model(inp, z, name="UE_Lightweight_Encoder")


# ============================================================
# BS-side temporal decoder
# ============================================================

def build_bs_temporal_decoder():
    """
    BS-side Doppler-aware temporal decoder.

    Inputs:
        latent_seq: [batch, SEQ_LEN, ENCODED_DIM]
        doppler_indicator: [batch, 1]

    Output:
        reconstructed CSI: [batch, 32, 32, 2]
    """
    latent_seq = Input(shape=(SEQ_LEN, ENCODED_DIM), name="bs_latent_sequence")
    doppler_indicator = Input(shape=(1,), name="doppler_indicator")

    # Temporal Convolution Network, using causal and dilated Conv1D.
    x = Conv1D(
        128,
        kernel_size=2,
        dilation_rate=1,
        padding="causal",
        activation="relu",
        name="tcn_dilation_1"
    )(latent_seq)

    x = Conv1D(
        128,
        kernel_size=2,
        dilation_rate=2,
        padding="causal",
        activation="relu",
        name="tcn_dilation_2"
    )(x)

    x = Conv1D(
        128,
        kernel_size=2,
        dilation_rate=4,
        padding="causal",
        activation="relu",
        name="tcn_dilation_4"
    )(x)

    x = GlobalAveragePooling1D(name="temporal_pooling")(x)

    # Doppler-aware conditioning.
    x = Concatenate(name="doppler_conditioning")([x, doppler_indicator])

    # CSI decoder.
    x = Dense(IMG_TOTAL, activation="linear", name="decoder_dense")(x)
    x = Reshape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="decoder_reshape")(x)

    for i in range(RESIDUAL_NUM):
        x = residual_decoder_block(x)

    out = Conv2D(
        IMG_CHANNELS,
        kernel_size=(3, 3),
        padding="same",
        activation="linear",
        name="reconstructed_csi"
    )(x)

    return Model(
        inputs=[latent_seq, doppler_indicator],
        outputs=out,
        name="BS_Doppler_Aware_TCN_Decoder"
    )


# ============================================================
# Full DA-LiteTCN CsiNet model
# ============================================================

def build_da_litetcn_csinet():
    """
    Full proposed architecture.

    Input:
        CSI sequence with shape [batch, SEQ_LEN, 32, 32, 2]

    Output:
        reconstructed current CSI with shape [batch, 32, 32, 2]

    Important:
        This version uses TimeDistributed and Lambda layers.
        It does NOT directly apply tf.stack to KerasTensor.
    """
    csi_seq = Input(
        shape=(SEQ_LEN, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        name="csi_sequence_input"
    )

    ue_encoder = build_lightweight_ue_encoder()
    bs_decoder = build_bs_temporal_decoder()

    # Apply the same UE encoder to each CSI frame.
    # Output shape: [batch, SEQ_LEN, ENCODED_DIM]
    latent_seq = TimeDistributed(
        ue_encoder,
        name="time_distributed_ue_encoder"
    )(csi_seq)

    # Doppler indicator:
    # d_t = ||z_t - z_{t-1}||_2
    # Lambda is used to make this compatible with Keras Functional API.
    doppler_indicator = Lambda(
        lambda z: tf.norm(
            z[:, -1, :] - z[:, -2, :],
            ord=2,
            axis=1,
            keepdims=True
        ),
        name="doppler_indicator_from_latent_difference"
    )(latent_seq)

    reconstructed_csi = bs_decoder([latent_seq, doppler_indicator])

    model = Model(
        inputs=csi_seq,
        outputs=reconstructed_csi,
        name="DA_LiteTCN_CsiNet"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    return model


# ============================================================
# Main: build model and run dummy test
# ============================================================

if __name__ == "__main__":
    model = build_da_litetcn_csinet()
    model.summary()

    dummy_input = np.random.randn(
        2,
        SEQ_LEN,
        IMG_HEIGHT,
        IMG_WIDTH,
        IMG_CHANNELS
    ).astype("float32")

    dummy_output = model.predict(dummy_input)

    print("Dummy input shape:", dummy_input.shape)
    print("Dummy output shape:", dummy_output.shape)