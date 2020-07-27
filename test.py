import sys
sys.settrace

import AEGEAN as AG
import numpy as np
# import os
# PID, HOST = os.getpid(), os.uname()[1]

experiments = {}
experiments['AEGEAN_128'] = [('img_size', 128), ]
# experiments['Simpsons_128'] = [('datapath', '../database/Simpsons-Face_clear/cp/'), ('img_size', 128), ('n_epochs', 128), ('do_SSIM', False)]
# experiments['AEGEAN_64'] = [('img_size', 64)]# , ('n_epochs', 512)]
# experiments['Simpsons_64'] = [('datapath', '../database/Simpsons-Face_clear/cp/'), ('img_size', 64), ('do_SSIM', True)]
# experiments['AEGEAN_256'] = [('img_size', 256), ]
# experiments['Simpsons_256'] = [('datapath', '../database/Simpsons-Face_clear/cp/'), ('img_size', 256)] #, ('do_SSIM', False)]
# experiments['butterflies_256'] = [('datapath', '../database/swapnesh_butterflies/'), ('img_size', 256)]
# experiments['butterflies_64'] = [('datapath', '../database/swapnesh_butterflies/'), ('img_size', 64)]
# experiments['butterflies_256'] = [('datapath', '../database/swapnesh_butterflies/'), ('img_size', 256)]
# experiments['clouds'] = [('datapath', '../database/downloads/cloud/'), ('img_size', 256)]
# experiments['Simpsons_64'] = [('datapath', '../database/Simpsons-Face_clear/cp/'), ('img_size', 64), ('n_epochs', 64), ('do_SSIM', False)]
#
# experiments['AE'] = [('lrG', 0.), ('img_size', 64), ('n_epochs', 64)] # still training the discriminator but G is not supervised by D
# experiments['AE'] = [('lrG', 0.), ('lrD', 0.)]
# experiments['Holidays'] = [('datapath', '../../../../quantic/Photos/2019'), ('img_size', 128)]

do_test = True

for expname in experiments.keys():
    def init():
        opt = AG.init()
        for variable, value in experiments[expname]:
            vars(opt)[variable] = value
        tag = f'{expname}_' #{opt.img_size}_'
        return tag, opt

    # VANILLA
    tag, opt = init()
    opt.run_path = tag + 'vanilla'
    print(opt)
    AG.learn(opt)

    # Does it help a GAN to be coupled with an AE ?
    tag, opt = init()
    opt.do_joint = not opt.do_joint
    if opt.do_joint:
        opt.run_path = tag + 'do_joint'
    else:
        opt.do_insight = False
        opt.run_path = tag + 'no_joint'
    AG.learn(opt)

    # What if the discriminator has only acces to the image reconstructed by the AE ?
    tag, opt = init()
    opt.do_insight = not opt.do_insight
    if opt.do_insight:
        opt.run_path = tag + 'do_insight'
    else:
        opt.run_path = tag + 'no_insight'
    AG.learn(opt)

    if True:

        # GAN_losses = ['original', 'wasserstein', 'ian', 'alternative', 'alternativ2']
        GAN_losses = ['original', 'ian',  'hinge', 'wasserstein', 'alternative', 'alternativ2', 'alternativ3']
        GAN_losses = ['original',  'hinge', 'alternativ3']
        # GAN_losses = ['original', 'ian', 'alternativ3']

        for GAN_loss in GAN_losses:
            tag, opt = init()
            if opt.lrD > 0:
                opt.run_path = tag + 'GAN_loss_' + GAN_loss
                opt.GAN_loss = GAN_loss
                opt.run_path += '_no_bn'
                opt.bn_eps = np.inf
                AG.learn(opt)

        #GAN_losses.remove(opt.GAN_loss)
        for GAN_loss in GAN_losses:
            tag, opt = init()
            if opt.lrD > 0:
                opt.run_path = tag + 'GAN_loss_' + GAN_loss
                opt.GAN_loss = GAN_loss
                AG.learn(opt)

    for padding_mode in ['reflect', 'border', 'zeros']: # https://pytorch.org/docs/1.4.0/nn.functional.html#grid-sample
        tag, opt = init()
        opt.run_path = tag + 'padding_mode_' + padding_mode
        opt.padding_mode = padding_mode
        AG.learn(opt)

    for optimizer in ['sgd', 'adam', 'rmsprop']: # https://pytorch.org/docs/1.4.0/nn.functional.html#grid-sample
        tag, opt = init()
        opt.run_path = tag + 'optimizer_' + optimizer
        opt.optimizer = optimizer
        AG.learn(opt)

    base = 2

    tag, opt = init()
    opt.run_path = tag + 'low_batch_size'
    opt.batch_size //= base
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'high_batch_size'
    opt.batch_size *= base
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'small_lrE'
    opt.lrE /= base
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'big_lrE'
    opt.lrE *= base
    AG.learn(opt)

    tag, opt = init()
    if opt.lrD > 0:
        opt.run_path = tag + 'small_lrD'
        opt.lrD /= base
        AG.learn(opt)

    tag, opt = init()
    if opt.lrD > 0:
        opt.run_path = tag + 'big_lrD'
        opt.lrD *= base
        AG.learn(opt)

    tag, opt = init()
    if opt.lrG > 0:
        opt.run_path = tag + 'small_lrG'
        opt.lrG /= base
        AG.learn(opt)

    tag, opt = init()
    if opt.lrG > 0:
        opt.run_path = tag + 'big_lrG'
        opt.lrG *= base
        AG.learn(opt)

    if False:
        tag, opt = init()
        opt.gamma = .618 if opt.gamma==1. else 1.
        # opt.gamma = 1.
        opt.run_path = tag + f'gamma_{str(opt.gamma)}'
        AG.learn(opt)

        tag, opt = init()
        opt.gamma = 1.4 if opt.gamma==1. else 1.
        # opt.gamma = 1.
        opt.run_path = tag + f'gamma_{str(opt.gamma)}'
        AG.learn(opt)

    if True:
        tag, opt = init()
        opt.run_path = tag + 'small_window_size'
        opt.window_size //= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'big_window_size'
        opt.window_size *= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'no_channel0_bg'
        opt.channel0_bg = 0
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'small_channel0_bg'
        opt.channel0_bg = 2
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'big_channel0_bg'
        opt.channel0_bg = opt.channel0 // 4
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'small_channel0'
        opt.channel0 //= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'big_channel0'
        opt.channel0 *= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'small_channel1'
        opt.channel1 //= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'big_channel1'
        opt.channel1 *= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'small_channel2'
        opt.channel2 //= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'big_channel2'
        opt.channel2 *= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'small_channel3'
        opt.channel3 //= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'big_channel3'
        opt.channel3 *= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'small_channel4'
        opt.channel4 //= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'big_channel4'
        opt.channel4 *= base
        AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'no_E_noise'
    opt.E_noise = 0.
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'low_E_noise'
    opt.E_noise /= base
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'high_E_noise'
    opt.E_noise *= base
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'no_D_noise'
    opt.D_noise = 0.
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'low_D_noise'
    opt.D_noise /= base
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'high_D_noise'
    opt.D_noise *= base
    AG.learn(opt)
    tag, opt = init()
    opt.run_path = tag + 'no_G_noise'
    opt.G_noise = 0.
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'low_G_noise'
    opt.G_noise /= base
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'high_G_noise'
    opt.G_noise *= base
    AG.learn(opt)

    tag, opt = init()
    opt.do_SSIM = not opt.do_SSIM
    if opt.do_SSIM:
        opt.run_path = tag + 'do_SSIM'
    else:
        opt.run_path = tag + 'no_SSIM'
    AG.learn(opt)


    if do_test:

        tag, opt = init()
        opt.init_weight = not opt.init_weight
        if opt.init_weight:
            opt.run_path = tag + 'do_init_weight'
        else:
            opt.run_path = tag + 'no_init_weight'
        AG.learn(opt)

        tag, opt = init()
        opt.do_bias = not opt.do_bias
        if opt.do_bias:
            opt.run_path = tag + 'do_bias'
        else:
            opt.run_path = tag + 'no_bias'
        AG.learn(opt)


        tag, opt = init()
        opt.run_path = tag + 'no_dropout'
        opt.dropout = 0.
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'low_dropout'
        opt.dropout = 0.001
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'high_dropout'
        opt.dropout = 0.01
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'zero_beta1'
        opt.beta1 = 0.0
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'low_beta1'
        opt.beta1 = 0.3
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'high_beta1'
        opt.beta1 = 0.925
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'low_beta2'
        opt.beta2 = 0.8
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'high_beta2'
        opt.beta2 = 0.95
        AG.learn(opt)

        tag, opt = init()
        if opt.rand_affine == 0.:
            opt.run_path = tag + 'do_affine'
            opt.rand_affine = 2.
        else:
            opt.run_path = tag + 'no_affine'
            opt.rand_affine = 0.
        AG.learn(opt)

        tag, opt = init()
        opt.lrelu = 0.1 if opt.lrelu==0. else 0.
        opt.run_path = tag + 'relu' if opt.lrelu==0. else tag + 'lrelu'
        AG.learn(opt)


        tag, opt = init()
        if opt.lrD > 0:
            opt.run_path = tag + 'high_valid_smooth'
            opt.valid_smooth = 0.99
            AG.learn(opt)
        tag, opt = init()
        if opt.lrD > 0:
            opt.run_path = tag + 'low_valid_smooth'
            opt.valid_smooth = 0.9
            AG.learn(opt)
        tag, opt = init()
        if opt.lrD > 0:
            opt.run_path = tag + 'no_valid_smooth'
            opt.valid_smooth = 1.
            AG.learn(opt)

    #
    # tag, opt = init()
    # if opt.bn_eps == np.inf:
    #     opt.run_path = tag + 'do_bn'
    #     opt.bn_eps = .3
    # else:
    #     opt.run_path = tag + 'no_bn'
    #     opt.bn_eps = np.inf
    # AG.learn(opt)
    base = 8

    # what's the effect of a smaller latent_dim ?
    tag, opt = init()
    opt.latent_dim, opt.run_path = opt.latent_dim//base, tag + 'small_latent_dim'
    AG.learn(opt)

    # what's the effect of a smaller latent_dim ?
    tag, opt = init()
    opt.latent_dim, opt.run_path = opt.latent_dim*base, tag + 'large_latent_dim'
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'small_lambdaE'
    opt.lambdaE /= base
    AG.learn(opt)

    tag, opt = init()
    opt.run_path = tag + 'big_lambdaE'
    opt.lambdaE *= base
    AG.learn(opt)

    if do_test:

        tag, opt = init()
        opt.run_path = tag + 'small_bn_eps'
        opt.bn_eps /= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'big_bn_eps'
        opt.bn_eps *= base
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'small_bn_momentum'
        opt.bn_momentum = .1
        AG.learn(opt)

        tag, opt = init()
        opt.run_path = tag + 'big_bn_momentum'
        opt.bn_momentum = .9
        AG.learn(opt)
