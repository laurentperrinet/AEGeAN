import AEGEAN as AG
import numpy as np
# import os
# PID, HOST = os.getpid(), os.uname()[1]
base = 4

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
# tag = f'AE_{HOST}_{opt.img_size}_'
tag = f'AE_{opt.img_size}_'

# VANILLA
opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'vanilla'
print(opt)
AG.learn(opt)

# what's the effect of a smaller latent_dim ?
opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.latent_dim, opt.runs_path = opt.latent_dim//2, tag + 'small_latent'
AG.learn(opt)

# what's the effect of a smaller latent_dim ?
opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.latent_dim, opt.runs_path = opt.latent_dim*2, tag + 'large_latent'
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'small_lrE'
opt.lrE /= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'big_lrE'
opt.lrE *= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'small_lambdaE'
opt.lambdaE /= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'big_lambdaE'
opt.lambdaE *= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'small_channel0'
opt.channel0 //= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'big_channel0'
opt.channel0 *= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'small_channel1'
opt.channel1 //= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'big_channel1'
opt.channel1 *= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'small_channel2'
opt.channel2 //= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'big_channel2'
opt.channel2 *= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'low_D_noise'
opt.D_noise /= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'high_D_noise'
opt.D_noise *= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
print('DEBUG opt.D_noise = ', opt.D_noise)
if opt.do_whitening:
    opt.runs_path = tag + 'no_whitening'
else:
    opt.runs_path = tag + 'do_whitening'
opt.do_whitening = not opt.do_whitening
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
if opt.init_weight:
    opt.runs_path = tag + 'no_init_weight'
else:
    opt.runs_path = tag + 'do_init_weight'
opt.init_weight = not opt.init_weight
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'low_batch_size'
opt.batch_size //= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'high_batch_size'
opt.batch_size *= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'low_adam_beta1'
opt.beta1 = 0.3
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'low_adam_beta2'
opt.beta2 = 0.99
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'high_adam_beta1'
opt.beta1 = 0.99
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'high_adam_beta2'
opt.beta2 = 0.9999
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'no_affine'
opt.rand_affine = 0.
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'relu' if opt.lrelu==0. else tag + 'lrelu'
opt.lrelu = 0.02 if opt.lrelu==0. else 0.
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'valid_smooth'
opt.valid_smooth = 0.9
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
if opt.bn_eps == np.inf:
    opt.runs_path = tag + 'do_bn'
    opt.bn_eps = .1
else:
    opt.runs_path = tag + 'no_bn'
    opt.bn_eps = np.inf
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'small_eps'
opt.bn_eps /= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'big_eps'
opt.bn_eps *= base
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'small_momentum'
opt.bn_momentum = .01
AG.learn(opt)

opt = AG.init()
opt.lrG, opt.lrD = 0., 0.
opt.runs_path = tag + 'big_momentum'
opt.bn_momentum = .5
AG.learn(opt)
