#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from .train import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen
import numpy as np
import sys
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# import tensorflow as tf

sq2 = np.sqrt(2)
sq10 = np.sqrt(10)
sq42 = np.sqrt(42)


def nle(mu, mean, var, thre):
    # ext_probs = np.zeros((ule.shape[0], 2 ** (mu // 2)))
    if mu == 2:  # {-1,+1}
        P0 = tf.maximum(tf.exp(-tf.square(-1 / sq2 - mean) / (2 * var)), thre)  # (bs, 2N, 1)
        P1 = tf.maximum(tf.exp(-tf.square(1 / sq2 - mean) / (2 * var)), thre)
        u_post = (P1 - P0) / (P1 + P0) / sq2
        v_post = (P0 * tf.square(u_post + 1 / sq2) + P1 * tf.square(u_post - 1 / sq2)) / (P1 + P0)
        ext_probs = tf.concat([P0, P1], axis=2)
    elif mu == 4:  # {-3,-1,+1,+3}
        P_3 = tf.maximum(tf.exp(-tf.square(-3 / sq10 - mean) / (2 * var)), thre)
        P_1 = tf.maximum(tf.exp(-tf.square(-1 / sq10 - mean) / (2 * var)), thre)
        P1 = tf.maximum(tf.exp(-tf.square(1 / sq10 - mean) / (2 * var)), thre)
        P3 = tf.maximum(tf.exp(-tf.square(3 / sq10 - mean) / (2 * var)), thre)
        u_post = (-3 * P_3 - P_1 + P1 + 3 * P3) / (P_3 + P_1 + P1 + P3) / sq10
        v_post = (P_3 * tf.square(u_post + 3 / sq10) + P_1 * tf.square(u_post + 1 / sq10) +
                  P1 * tf.square(u_post - 1 / sq10) + P3 * tf.square(u_post - 3 / sq10)) / (P_3 + P_1 + P1 + P3)
        ext_probs = tf.concat([P_3, P_1, P3, P1], axis=2)  # order corresponds to mapping
    else:  # {-1,+1}
        P_7 = tf.maximum(tf.exp(-tf.square(-7 / sq42 - mean) / (2 * var)), thre)
        P_5 = tf.maximum(tf.exp(-tf.square(-5 / sq42 - mean) / (2 * var)), thre)
        P_3 = tf.maximum(tf.exp(-tf.square(-3 / sq42 - mean) / (2 * var)), thre)
        P_1 = tf.maximum(tf.exp(-tf.square(-1 / sq42 - mean) / (2 * var)), thre)
        P1 = tf.maximum(tf.exp(-tf.square(1 / sq42 - mean) / (2 * var)), thre)
        P3 = tf.maximum(tf.exp(-tf.square(3 / sq42 - mean) / (2 * var)), thre)
        P5 = tf.maximum(tf.exp(-tf.square(5 / sq42 - mean) / (2 * var)), thre)
        P7 = tf.maximum(tf.exp(-tf.square(7 / sq42 - mean) / (2 * var)), thre)
        u_post = (-7 * P_7 - 5 * P_5 - 3 * P_3 - P_1 + P1 + 3 * P3 + 5 * P5 + 7 * P7) / (
                P_7 + P_5 + P_3 + P_1 + P1 + P3 + P5 + P7) / sq42
        v_post = (P_7 * tf.square(u_post + 7 / sq42) + P_5 * tf.square(u_post + 5 / sq42) +
                  P_3 * tf.square(u_post + 3 / sq42) + P_1 * tf.square(u_post + 1 / sq42) +
                  P1 * tf.square(u_post - 1 / sq42) + P3 * tf.square(u_post - 3 / sq42) +
                  P5 * tf.square(u_post - 5 / sq42) + P7 * tf.square(u_post - 7 / sq42)) / \
                 (P_7 + P_5 + P_3 + P_1 + P1 + P3 + P5 + P7)
        ext_probs = tf.concat([P_7, P_5, P_1, P_3, P7, P5, P1, P3], axis=2)
    return u_post, v_post, ext_probs


def CG(u, p, residual, r_norm, XI, sample_size):
    # compute the approximate solution based on prior conjugate direction and residual
    XI_p = tf.matmul(XI, p)  # bs*2M*1
    a = r_norm / tf.matmul(tf.transpose(p, perm=[0, 2, 1]), XI_p)
    u = tf.add(u, a * p)
    # compute conjugate direction and residual
    residual = tf.add(residual, -a * XI_p)
    r_norm_last = r_norm
    r_norm = tf.reshape(tf.square(tf.norm(residual, axis=(1, 2))), [sample_size, 1, 1])
    # r_norm_last = tf.maximum(r_norm_last,tf.constant(1e-20))
    b = r_norm / r_norm_last
    p = tf.add(residual, b * p)
    # r_norm = tf.maximum(r_norm, tf.constant(1e-20))
    return u, p, residual, r_norm


def build_EP(trainSet):
    
    T = trainSet.T
    Mr, Nt, mu, SNR = trainSet.Mr, trainSet.Nt, trainSet.mu, trainSet.snr
    lr, maxit = trainSet.lr, trainSet.maxit
    vsample_size = trainSet.vsample_size
    total_batch, batch_size = trainSet.total_batch, trainSet.batch_size
    savefile = trainSet.savefile
    
    prob, test = trainSet.prob, trainSet.test
    
    layers = []
    # layerinfo: (name, xhat, newvars)
    
    H = prob.H_      # (bs, 2M, 2N)
    x = prob.x_      # (bs, 2N, 1)
    y = prob.y_      # (bs, 2M, 1)
    sigma2 = prob.sigma2_ # (bs, 1, 1)
    sample_size = prob.sample_size_ # bs
    
    HT = tf.transpose(H, perm=[0, 2, 1])
    HTH = tf.matmul(HT, H)
    noise_var = sigma2 / 2
    
    def inv_sigmoid(y):
        x = np.log(y / (1 - y))
        return x
        
    # Precompute some tensorflow constants
    eps = tf.constant(5e-7, dtype=tf.float32)
    pth = tf.constant(1e-100, dtype=tf.float64)
    Lambda = 1 / (0.5 * tf.ones_like(x, dtype=tf.float32))
    gamma = tf.zeros_like(x, tf.float32)
    
    for t in range(T):
        # Learnable damping factor for each layer
        beta = tf.Variable(float(inv_sigmoid(min(0.1 * np.exp(t / 1.5), 0.7))), dtype=tf.float32, name='beta_' + str(t))
        
        # 1. Compute the mean and covariance matrix
        # Sigma: (bs, 2N, 2N)
        Sigma = tf.linalg.inv(HTH / noise_var + tf.matrix_diag(tf.squeeze(Lambda)))
        # Mu: (bs, 2N, 1)
        Mu = tf.matmul(Sigma, tf.matmul(HT, y) / noise_var + gamma)
        
        # 2. Compute the extrinsic mean and covariance matrix
        diag = tf.expand_dims(tf.matrix_diag_part(Sigma), -1) # (bs, 2N, 1)
        vab = tf.divide(1, tf.divide(1, diag) - Lambda) # (bs, 2N, 1)
        vab = tf.maximum(vab, eps)
        uab = vab * (Mu / diag - gamma) # (bs, 2N, 1)
        
        # 3. Compute the posterior mean and covariance matrix
        uab_64 = tf.cast(uab, dtype=tf.float64)
        vab_64 = tf.cast(vab, dtype=tf.float64)
        
        # 呼叫你上方定義好的 nle 函數 (Non-linear Estimator)
        ub, vb, _ = nle(mu, uab_64, vab_64, pth)
        
        ub = tf.cast(ub, dtype=tf.float32)
        vb = tf.maximum(tf.cast(vb, dtype=tf.float32), 0.1 * eps)
        
        # 4. Moment matching and damping
        gamma_last = gamma
        Lambda_last = Lambda
        
        gamma = ub / vb - uab / vab
        Lambda = 1 / vb - 1 / vab
        
        gamma = tf.math.sigmoid(beta) * gamma + (1 - tf.math.sigmoid(beta)) * gamma_last
        Lambda = tf.math.sigmoid(beta) * Lambda + (1 - tf.math.sigmoid(beta)) * Lambda_last
        
        layers.append(('EP T={0}'.format(t), uab, (beta,)))
        
    loss = tf.nn.l2_loss(uab - x)
    lr_ = tf.Variable(lr, name='lr', trainable=False)
    
    if tf.trainable_variables() is not None:
        train = tf.train.AdamOptimizer(lr_).minimize(loss, var_list=tf.trainable_variables())
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    # 載入預訓練參數 (若有)
    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))
    
    for name, uab, var_list in layers:
        if name not in done:
            if len(var_list):
                describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
            else:
                describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables()])
            print(name + ' ' + describe_var_list)
            done = np.append(done, name)
            
    print(log)
    
    if test:
        return sess, uab
        
    loss_history = []
    save = {} # for the best model
    ivl = 1
    
    yval, xval, Hval, sigma2val, _, _, _ = sample_gen(trainSet, 1, vsample_size)
    
    for i in range(maxit + 1):
        y_, x_, H_, sigma2_, _, _, _ = sample_gen(trainSet, batch_size * total_batch, 1)
        
        if i % ivl == 0:
            # validation: don't use optimizer
            loss_val = sess.run(loss, feed_dict={
                prob.y_: yval,
                prob.x_: xval,
                prob.H_: Hval,
                prob.sigma2_: sigma2val,
                prob.sample_size_: vsample_size
            })
            
            if np.isnan(loss_val):
                raise RuntimeError('loss is NaN')
                
            loss_history = np.append(loss_history, loss_val)
            loss_best = loss_history.min()
            
            if loss_val == loss_best:
                for v in tf.trainable_variables():
                    save[str(v.name)] = sess.run(v)
                    
            sys.stdout.write('\ri={i:<6d} loss={loss:.9f} (best={best:.9f})'.format(i=i, loss=loss_val, best=loss_best))
            sys.stdout.flush()
            
        if i % (100 * ivl) == 0:
            print('')
            
        for m in range(total_batch):
            sess.run(train, feed_dict={
                prob.y_: y_[m * batch_size:(m + 1) * batch_size],
                prob.x_: x_[m * batch_size:(m + 1) * batch_size],
                prob.H_: H_[m * batch_size:(m + 1) * batch_size],
                prob.sigma2_: sigma2_[m * batch_size:(m + 1) * batch_size],
                prob.sample_size_: batch_size
            })
            
    # Restore the best model
    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k + ' = ' + str(d))
            
    log = log + '\nloss={loss:.9f} in {i} iterations best={best:.9f} in {j} iterations'.format(
        loss=loss_val, i=i, best=loss_best, j=loss_history.argmin())
        
    state['done'] = done
    state['log'] = log
    save_trainable_vars(sess, savefile, **state)

    return sess, uab
