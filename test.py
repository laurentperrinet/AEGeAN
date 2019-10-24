import AEGEAN as AG
import numpy as np
# import os
# PID, HOST = os.getpid(), os.uname()[1]

experiments = {}
experiments['AEGEAN'] = []
# experiments['AE'] = [('lrG', 0.), ('lrD', 0.)]
experiments['AE'] = [('lrG', 0.)] # still training the discriminator but G is not supervised by D
experiments['Simpsons'] = [('datapath', '../database/Simpsons-Face_clear/cp/')]

for expname in experiments.keys():
    def init():
        opt = AG.init()
        for variable, value in experiments[expname]:
            vars(opt)[variable] = value
        tag = f'{expname}_{opt.img_size}_'
        return tag, opt

    # VANILLA
    tag, opt = init()
    opt.runs_path = tag + 'vanilla'
    print(opt)
    AG.learn(opt)


    base = 2

    GAN_losses = ['original', 'wasserstein', 'ian', 'alternative']
    for GAN_loss in GAN_losses:
        tag, opt = init()
        if opt.lrD > 0:
            opt.runs_path = tag + 'GAN_loss_' + GAN_loss
            opt.GAN_loss = GAN_loss
            opt.runs_path += '_no_bn'
            opt.bn_eps = np.inf
            AG.learn(opt)

    GAN_losses.remove(opt.GAN_loss)

    for GAN_loss in GAN_losses:
        tag, opt = init()
        if opt.lrD > 0:
            opt.runs_path = tag + 'GAN_loss_' + GAN_loss
            opt.GAN_loss = GAN_loss
            AG.learn(opt)


    # what's the effect of a smaller latent_dim ?
    tag, opt = init()
    opt.latent_dim, opt.runs_path = opt.latent_dim//base, tag + 'small_latent'
    AG.learn(opt)

    # what's the effect of a smaller latent_dim ?
    tag, opt = init()
    opt.latent_dim, opt.runs_path = opt.latent_dim*base, tag + 'large_latent'
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'small_lrE'
    opt.lrE /= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'big_lrE'
    opt.lrE *= base
    AG.learn(opt)

    tag, opt = init()
    if opt.lrD > 0:
        opt.runs_path = tag + 'small_lrD'
        opt.lrD /= base
        AG.learn(opt)

    tag, opt = init()
    if opt.lrD > 0:
        opt.runs_path = tag + 'big_lrD'
        opt.lrD *= base
        AG.learn(opt)

    tag, opt = init()
    if opt.lrG > 0:
        opt.runs_path = tag + 'small_lrG'
        opt.lrG /= base
        AG.learn(opt)

    tag, opt = init()
    if opt.lrG > 0:
        opt.runs_path = tag + 'big_lrG'
        opt.lrG *= base
        AG.learn(opt)


    tag, opt = init()
    opt.runs_path = tag + 'small_window_size'
    opt.window_size //= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'big_window_size'
    opt.window_size *= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'small_channel0'
    opt.channel0 //= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'big_channel0'
    opt.channel0 *= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'small_channel1'
    opt.channel1 //= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'big_channel1'
    opt.channel1 *= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'small_channel2'
    opt.channel2 //= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'big_channel2'
    opt.channel2 *= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'no_D_noise'
    opt.D_noise = 0.
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'low_D_noise'
    opt.D_noise /= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'high_D_noise'
    opt.D_noise *= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'no_G_noise'
    opt.G_noise = 0.
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'low_G_noise'
    opt.G_noise /= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'high_G_noise'
    opt.G_noise *= base
    AG.learn(opt)

    tag, opt = init()
    opt.do_SSIM = not opt.do_SSIM
    if opt.do_SSIM:
        opt.runs_path = tag + 'do_SSIM'
    else:
        opt.runs_path = tag + 'no_SSIM'
    AG.learn(opt)

    tag, opt = init()
    opt.init_weight = not opt.init_weight
    if opt.init_weight:
        opt.runs_path = tag + 'do_init_weight'
    else:
        opt.runs_path = tag + 'no_init_weight'
    AG.learn(opt)

    tag, opt = init()
    opt.do_bias = not opt.do_bias
    if opt.do_bias:
        opt.runs_path = tag + 'do_bias'
    else:
        opt.runs_path = tag + 'no_bias'
    AG.learn(opt)

    if False:
        tag, opt = init()
        opt.runs_path = tag + 'low_batch_size'
        opt.batch_size //= base
        AG.learn(opt)

        tag, opt = init()
        opt.runs_path = tag + 'high_batch_size'
        opt.batch_size *= base
        AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'low_beta1'
    opt.beta1 = 0.5
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'high_beta1'
    opt.beta1 = 0.9
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'low_beta2'
    opt.beta2 = 0.9
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'high_beta2'
    opt.beta2 = 0.99
    AG.learn(opt)

    tag, opt = init()
    if opt.rand_affine == 0.:
        opt.runs_path = tag + 'do_affine'
        opt.rand_affine = 2.
    else:
        opt.runs_path = tag + 'no_affine'
        opt.rand_affine = 0.
    AG.learn(opt)

    tag, opt = init()
    opt.lrelu = 0.1 if opt.lrelu==0. else 0.
    opt.runs_path = tag + 'relu' if opt.lrelu==0. else tag + 'lrelu'
    AG.learn(opt)


    tag, opt = init()
    if opt.lrD > 0:
        opt.runs_path = tag + 'low_valid_smooth'
        opt.valid_smooth = 0.9
        AG.learn(opt)
    tag, opt = init()
    if opt.lrD > 0:
        opt.runs_path = tag + 'no_valid_smooth'
        opt.valid_smooth = 1.
        AG.learn(opt)

    tag, opt = init()
    opt.do_joint = not opt.do_joint
    if opt.do_joint:
        opt.runs_path = tag + 'do_joint'
    else:
        opt.runs_path = tag + 'no_joint'
    AG.learn(opt)

    #
    # tag, opt = init()
    # if opt.bn_eps == np.inf:
    #     opt.runs_path = tag + 'do_bn'
    #     opt.bn_eps = .3
    # else:
    #     opt.runs_path = tag + 'no_bn'
    #     opt.bn_eps = np.inf
    # AG.learn(opt)
    base = 8

    tag, opt = init()
    opt.runs_path = tag + 'small_lambdaE'
    opt.lambdaE /= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'big_lambdaE'
    opt.lambdaE *= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'small_bn_eps'
    opt.bn_eps /= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'big_bn_eps'
    opt.bn_eps *= base
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'small_bn_momentum'
    opt.bn_momentum = .3
    AG.learn(opt)

    tag, opt = init()
    opt.runs_path = tag + 'big_bn_momentum'
    opt.bn_momentum = .9
    AG.learn(opt)
