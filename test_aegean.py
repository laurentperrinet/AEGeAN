import os
PID, HOST = os.getpid(), os.uname()[1]

import numpy as np
import AEGEAN as AG

opt = AG.init()
tag = f'{HOST}_{opt.img_size}_'

#
# for G_loss in ['wasserstein', 'ian', 'alternative', 'original']:
#     opt = AG.init()
#     opt.runs_path = tag + 'G_loss_' + G_loss
#     opt.G_loss = G_loss
#     AG.learn(opt)
#
# opt = AG.init()
# opt.runs_path = tag + 'do_whitening'
# opt.do_whitening = True
# AG.learn(opt)
#
#
# # what's the effect of a smaller latent_dim ?
# opt = AG.init()
# opt.latent_dim, opt.runs_path = 4, tag + 'small_latent'
# AG.learn(opt)
#
# # what's the effect of a smaller latent_dim ?
# opt = AG.init()
# opt.latent_dim, opt.runs_path = 54, tag + 'large_latent'
# AG.learn(opt)

#
# opt = AG.init()
# opt.runs_path = tag + 'small_channel0'
# opt.channel0 //= 2
# AG.learn(opt)
#
# opt = AG.init()
# opt.runs_path = tag + 'big_channel0'
# opt.channel0 *= 2
# AG.learn(opt)
#
# opt = AG.init()
# opt.runs_path = tag + 'small_channel1'
# opt.channel1 //= 2
# AG.learn(opt)
#
# opt = AG.init()
# opt.runs_path = tag + 'big_channel1'
# opt.channel1 *= 2
# AG.learn(opt)
#
# opt = AG.init()
# opt.runs_path = tag + 'small_channel2'
# opt.channel2 //= 2
# AG.learn(opt)
#
# opt = AG.init()
# opt.runs_path = tag + 'big_channel2'
# opt.channel2 *= 2
# AG.learn(opt)
#
#

opt = AG.init()
opt.runs_path = tag + 'D_noise'
opt.D_noise = 0.5
AG.learn(opt)

opt = AG.init()
opt.runs_path = tag + 'no_affine'
opt.rand_affine = 0.
AG.learn(opt)


opt = AG.init()
opt.runs_path = tag + 'relu'
opt.lrelu = 0.
AG.learn(opt)

opt = AG.init()
opt.runs_path = tag + 'valid_smooth'
opt.valid_smooth = 0.9
AG.learn(opt)

# VANILLA
opt = AG.init()
opt.runs_path = tag + 'vanilla'
print(opt)
AG.learn(opt)


opt = AG.init()
opt.runs_path = tag + 'no_bn'
opt.bn_eps = np.inf
AG.learn(opt)

opt = AG.init()
opt.runs_path = tag + 'small_eps'
opt.bn_eps = 1e-10
AG.learn(opt)

opt = AG.init()
opt.runs_path = tag + 'big_eps'
opt.bn_eps = 1e-2
AG.learn(opt)

opt = AG.init()
opt.runs_path = tag + 'small_momentum'
opt.bn_momentum = .01
AG.learn(opt)

opt = AG.init()
opt.runs_path = tag + 'big_momentum'
opt.bn_momentum = .5
AG.learn(opt)
