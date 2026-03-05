"""
Exercise 2.5: DeepMIMO Channel Estimation with GAN

This script demonstrates how to:
1. Generate/Load channel data using the DeepMIMO dataset (Scenario: O1_60).
2. Train a Conditional GAN (CGAN) to model the channel distribution.

DeepMIMO Configuration (Scenario O1_60):
- Scenario: 'O1_60'
- Active Base Stations: BS 1
- Active Users: Rows 1-5 (User 1 to 5)
- Number of Antennas (BS): 64 (8x8 UPA)
- Number of Antennas (UE): 1
- Frequency: 60 GHz
- Bandwidth: 0.5 GHz
- OFDM Subcarriers: 1 (Narrowband assumption for this exercise, or extract central subcarrier)

TODO:
1. Configure DeepMIMO parameters in `generate_deepmimo_data` to generate the dataset.
   Required Scenario: 'O1_60', BS 1, Users 1-500, SISO configuration.
2. The GAN training part is provided (Exercise 2.4 logic).
"""

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.io as sio

# Try to import DeepMIMO if available. 
# Instructions: pip install DeepMIMOv3
try:
    import DeepMIMO
    DEEPMIMO_AVAILABLE = True
except ImportError:
    DEEPMIMO_AVAILABLE = False
    print("DeepMIMO library not found. Please install it: pip install DeepMIMO")

tf.disable_v2_behavior()
matplotlib.use('Agg')

# ────────────────────────  Configuration  ────────────────────────────────── #
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

tf.set_random_seed(100)
np.random.seed(100)

# ────────────────────────  DeepMIMO Data Generation  ─────────────────────── #
def generate_deepmimo_data():
    """
    Generates channel data for Scenario O1_60 using DeepMIMO v2.
    
    TODO:
        Perform the following steps:
        1. Set the correct dataset folder and scenario name ('O1_60').
        2. Configure DeepMIMO parameters:
           - Active BS: 1
           - Active Users: Rows 1 to 500 (to get ~100k samples or sufficient data)
           - Antennas: SISO configuration (1 BS antenna, 1 UE antenna)
           - OFDM: 1 subcarrier, 0.5 GHz bandwidth
        3. Generate data using DeepMIMO.generate_data(parameters).
        4. Extract the channel coefficients (h) for the SISO link.
        5. Flatten and return the channel data.

    Returns:
        h_dataset (np.ndarray): Flattened array of complex channel coefficients.
                               Shape: (num_samples, )
    """
    if not DEEPMIMO_AVAILABLE:
        if os.path.exists('DeepMIMO_dataset.mat'):
            print("Loading from existing DeepMIMO_dataset.mat...")
            mat_data = sio.loadmat('DeepMIMO_dataset.mat')
            return mat_data['h_dataset'].flatten()
        else:
            raise FileNotFoundError("DeepMIMO library not installed and 'DeepMIMO_dataset.mat' not found.")

    print("Generating data using DeepMIMO v2...")
    
    # TO DO: realize the function
    return h_dataset


# ────────────────────────  Model Definition (Same as Ex 2.4) ─────────────── #
def generator_conditional(z, conditioning): 
    """
    Build the Generator network for CGAN.
    """
    z_combine = tf.concat([z, conditioning], 1)
    G_h1 = tf.nn.relu(tf.matmul(z_combine, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_logit = tf.matmul(G_h3, G_W4) + G_b4
    return G_logit


def discriminator_conditional(X, conditioning):
    """
    Build the Discriminator network for CGAN.
    """
    z_combine = tf.concat([X, conditioning], 1)
    D_h1_real = tf.nn.relu(tf.matmul(z_combine / 4, D_W1) + D_b1)
    D_h2_real = tf.nn.relu(tf.matmul(D_h1_real, D_W2) + D_b2)
    D_h3_real = tf.nn.relu(tf.matmul(D_h2_real, D_W3) + D_b3)
    D_logit = tf.matmul(D_h3_real, D_W4) + D_b4
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


def sample_Z(sample_size):
    ''' Sampling the generation noise Z from normal distribution '''
    return np.random.normal(size=sample_size)


def xavier_init(size):
    """Xavier initialization."""
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def generate_real_samples_with_labels_DeepMIMO(h_dataset, number=100):
    """
    Generate real (labeled) samples for training the CGAN using DeepMIMO data.
    
    This function mirrors the logic of Exercise 2.4. 
    It combines the DeepMIMO channel coefficients with QAM symbols and noise
    to create training samples.
    """
    # 1. Randomly choose channel coefficients
    h_complex_for_plot = np.random.choice(h_dataset, number)
    
    # 2. Get real and imaginary parts
    h_r = np.real(h_complex_for_plot)
    h_i = np.imag(h_complex_for_plot)
    
    # 3. Generate Random QAM symbols
    # Define QAM constellation
    mean_set_QAM = np.asarray([-3-3j, -3-1j, -3+1j, -3+3j, -1-3j, -1-1j, -1+1j, -1+3j,
                               1-3j, 1-1j, 1+1j, 1+3j, 3-3j, 3-1j, 3+1j, 3+3j], dtype=np.complex64)
    # Random indices for QAM symbols
    labels_index = np.random.randint(0, len(mean_set_QAM), number)
    data_t = mean_set_QAM[labels_index]
    
    # 4. Simulate Received Signal (y = hx + n)
    h_complex = h_r + 1j * h_i
    transmit_data = h_complex * data_t
    
    # Add noise
    gaussion_random = np.random.multivariate_normal([0, 0], [[0.03, 0], [0, 0.03]], number).astype(np.float32)
    # y = hx + n
    received_data_complex = transmit_data + (gaussion_random[:,0] + 1j * gaussion_random[:,1])
    
    # Format received data for Network: [Real, Imag]
    received_data = np.hstack((np.real(received_data_complex).reshape(number, 1),
                               np.imag(received_data_complex).reshape(number, 1)))
                               
    # 5. Construct Conditioning Vector
    # [Re{x}, Im{x}, Re{h}, Im{h}] / Normalisation Factor
    conditioning = np.hstack(
        (np.real(data_t).reshape(number, 1), np.imag(data_t).reshape(number, 1),
         h_r.reshape(number, 1), h_i.reshape(number, 1))) / 3
         
    return received_data, conditioning


# ────────────────────────  Main Execution  ────────────────────────────────── #
if __name__ == "__main__":
    
    # 1. QAM Constellation
    mean_set_QAM = np.asarray([-3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j, -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
                               1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j, 3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j
                               ], dtype=np.complex64)

    # 2. Get Data
    try:
        h_dataset = generate_deepmimo_data()
        print(f"Loaded DeepMIMO dataset with {len(h_dataset)} samples.")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create Dummy Data for code structure validation if generation fails
        print("Creating dummy Rayleigh data for testing...")
        h_dataset = (np.random.normal(0, 1, 1000) + 1j * np.random.normal(0, 1, 1000)).flatten()

    # 3. Configuration
    batch_size = 512
    condition_depth = 2
    condition_dim = 4
    Z_dim = 16
    model_name = 'ChannelGAN_DeepMIMO_'
    data_size = 10000 

    # 4. Generate Training Data
    # Note: using `h_dataset` which now comes from DeepMIMO
    data, one_hot_labels = generate_real_samples_with_labels_DeepMIMO(h_dataset, data_size)

    # 5. Graph Construction
    D_W1 = tf.Variable(xavier_init([2 + condition_dim, 32]))
    D_b1 = tf.Variable(tf.zeros(shape=[32]))
    D_W2 = tf.Variable(xavier_init([32, 32]))
    D_b2 = tf.Variable(tf.zeros(shape=[32]))
    D_W3 = tf.Variable(xavier_init([32, 32]))
    D_b3 = tf.Variable(tf.zeros(shape=[32]))
    D_W4 = tf.Variable(xavier_init([32, 1]))
    D_b4 = tf.Variable(tf.zeros(shape=[1]))
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3, D_W4, D_b4]
    
    G_W1 = tf.Variable(xavier_init([Z_dim + condition_dim, 128]))
    G_b1 = tf.Variable(tf.zeros(shape=[128]))
    G_W2 = tf.Variable(xavier_init([128, 128]))
    G_b2 = tf.Variable(tf.zeros(shape=[128]))
    G_W3 = tf.Variable(xavier_init([128, 128]))
    G_b3 = tf.Variable(tf.zeros(shape=[128]))
    G_W4 = tf.Variable(xavier_init([128, 2]))
    G_b4 = tf.Variable(tf.zeros(shape=[2]))
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3, G_W4, G_b4]
    
    R_sample = tf.placeholder(tf.float32, shape=[None, 2])
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
    Condition = tf.placeholder(tf.float32, shape=[None, condition_dim])
    G_sample = generator_conditional(Z, Condition)
    
    D_prob_real, D_logit_real = discriminator_conditional(R_sample, Condition)
    D_prob_fake, D_logit_fake = discriminator_conditional(G_sample, Condition)

    # 6. Loss & Optimization
    D_loss = tf.reduce_mean(D_logit_fake) - tf.reduce_mean(D_logit_real)
    G_loss = -1 * tf.reduce_mean(D_logit_fake)
    lambdda = 5
    alpha = tf.random_uniform(shape=tf.shape(R_sample), minval=0., maxval=1.)
    differences = G_sample - R_sample
    interpolates = R_sample + (alpha * differences)
    _, D_inter = discriminator_conditional(interpolates, Condition)
    gradients = tf.gradients(D_inter, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
    D_loss += lambdda * gradient_penalty
    
    D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss, var_list=theta_G)

    # 7. Training Loop
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    save_fig_path = model_name + "images"
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    
    print("Start Training...")
    # Training Loop (simplified for brevity in starter)
    # The student is expected to run this to verify their data generation works with the GAN
    saver = tf.train.Saver()
    plot_every = 1000
    
    for it in range(10001): # Reduce iterations for starter testing
        start_idx = it * batch_size % data_size
        if start_idx + batch_size >= len(data):
            continue
            
        X_mb = data[start_idx:start_idx + batch_size, :]
        one_hot_labels_mb = one_hot_labels[start_idx:start_idx + batch_size, :]
        
        for d_idx in range(5):
            _, D_loss_curr = sess.run([D_solver, D_loss],
                                      feed_dict={R_sample: X_mb, Z: sample_Z((batch_size, Z_dim)),
                                                 Condition: one_hot_labels_mb})
        _, G_loss_curr = sess.run([G_solver, G_loss],
                                  feed_dict={R_sample: X_mb, Z: sample_Z((batch_size, Z_dim)),
                                             Condition: one_hot_labels_mb})

        if it % plot_every == 0:
            print(f"Iter: {it}, D_loss: {D_loss_curr:.4f}, G_loss: {G_loss_curr:.4f}")
            # Visualization code would go here (omitted for brevity)
