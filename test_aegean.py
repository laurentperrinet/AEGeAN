import os
import AEGEAN as AG

tag = 'test_'

# VANILLA
opt = AG.init()
opt.runs_path = tag + 'vanilla'
print(opt)
AG.learn(opt)

# what's the effect of a smaller latent_dim ?
opt = AG.init()
opt.latent_dim, opt.runs_path = 4, tag + 'small_latent'
AG.learn(opt)

# what's the effect of a smaller latent_dim ?
opt = AG.init()
opt.latent_dim, opt.runs_path = 54, tag + 'large_latent'
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
