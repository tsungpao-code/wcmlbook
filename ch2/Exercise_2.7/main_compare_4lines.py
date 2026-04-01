import numpy as np
import tensorflow.compat.v1 as tf
import scipy.io as sio
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)

from tools import networks, raputil

K = 64
mu = 2
SNR_train = [5, 10, 15, 20, 25, 30, 35, 40]
training_epochs = 100
batch_size = 50

def run_case(ce_type='dnn', cp_flag=True):
    MSE_T = []
    MSE_F = []

    for snr in SNR_train:
        print(f"\nRunning {ce_type}, CP={cp_flag}, SNR={snr}")

        if ce_type == 'dnn':
            sess, input_holder, output = networks.build_ce_dnn(
                K,
                snr,
                training_epochs=training_epochs,
                batch_size=batch_size,
                savefile='dnn_ce/CE_DNN_' + ('CPFREE_' if cp_flag is False else '') +
                         str(2 ** mu) + 'QAM_SNR_' + str(snr) + 'dB.npz',
                test_flag=True,
                cp_flag=cp_flag,
                nh1=500,
                nh2=250
            )

            mse_t, mse_f = raputil.test_ce(
                sess,
                input_holder,
                output,
                snr,
                est_type='dnn',
                CP_flag=cp_flag
            )

        elif ce_type == 'mmse':
            mse_t, mse_f = raputil.test_ce(
                None,
                None,
                None,
                snr,
                est_type='mmse',
                CP_flag=cp_flag
            )

        else:
            raise ValueError("ce_type must be 'dnn' or 'mmse'")

        MSE_T.append(mse_t)
        MSE_F.append(mse_f)
        tf.reset_default_graph()

    return np.array(MSE_T), np.array(MSE_F)


# 1. DNN with CP
MSE_T_dnn_cp, MSE_F_dnn_cp = run_case('dnn', True)

# 2. LMMSE with CP
MSE_T_mmse_cp, MSE_F_mmse_cp = run_case('mmse', True)

# 3. DNN without CP
MSE_T_dnn_nocp, MSE_F_dnn_nocp = run_case('dnn', False)

# 4. LMMSE without CP
MSE_T_mmse_nocp, MSE_F_mmse_nocp = run_case('mmse', False)

print("\nDNN with CP:", MSE_F_dnn_cp)
print("LMMSE with CP:", MSE_F_mmse_cp)
print("DNN without CP:", MSE_F_dnn_nocp)
print("LMMSE without CP:", MSE_F_mmse_nocp)

sio.savemat('compare_results_4lines.mat', {
    'SNR': np.array(SNR_train),
    'MSE_F_dnn_cp': MSE_F_dnn_cp,
    'MSE_F_mmse_cp': MSE_F_mmse_cp,
    'MSE_F_dnn_nocp': MSE_F_dnn_nocp,
    'MSE_F_mmse_nocp': MSE_F_mmse_nocp
})

plt.figure(figsize=(10, 6))
plt.semilogy(SNR_train, MSE_F_dnn_cp, 'o-', linewidth=2.5, markersize=8, label='DNN with CP')
plt.semilogy(SNR_train, MSE_F_mmse_cp, 's-', linewidth=2.5, markersize=8, label='LMMSE with CP')
plt.semilogy(SNR_train, MSE_F_dnn_nocp, 'o--', linewidth=2.5, markersize=8, label='DNN without CP')
plt.semilogy(SNR_train, MSE_F_mmse_nocp, 's--', linewidth=2.5, markersize=8, label='LMMSE without CP')

plt.xlabel('SNR (dB)', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.title('Exercise 2.7: SISO-OFDM Channel Estimation', fontsize=18)
plt.grid(True, which='both', linestyle=':')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(r'C:\Data-Driven\compare_results_4lines.png', dpi=300)
plt.show()
