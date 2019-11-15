import os
import numpy as np
from torch.autograd import Variable


from .init import init
from .utils import *
from .plot import *
from .models import *
from pytorch_msssim import NMSSSIM

import torchvision.transforms as transforms
from torchvision.utils import save_image

# from torch.utils.data import DataLoader
from torchvision import datasets

import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d
import os
import sys
import time
import datetime

#
# # see research/UniStellar/AlignAndStack/2019-06-27-B1c_linear_inverter-decorrelation.ipynb
# KW = torch.zeros(size=(1, 1, 3, 3))
# KW[0, 0, :, 0] = torch.Tensor([0, -1, 0])
# KW[0, 0, :, 1] = torch.Tensor([-1, 4, -1])
# KW[0, 0, :, 2] = torch.Tensor([0, -1, 0])
# # KW = KW * torch.ones(1, 3, 1, 1)
# KW = KW * torch.eye(3).view(3, 3, 1, 1)
# KW /= np.sqrt(4. * 3)
# Kinv = torch.zeros((1, 1, 3, 3))
# Kinv[0, 0, :, 0] = torch.Tensor([.75, 1.5, .75])
# Kinv[0, 0, :, 1] = torch.Tensor([1.5, 4.5, 1.5])
# Kinv[0, 0, :, 2] = torch.Tensor([.75, 1.5, .75])
# # Kinv = Kinv * torch.ones(1, 3, 1, 1)
# Kinv = Kinv * torch.eye(3).view(3, 3, 1, 1)
# Kinv /= np.sqrt(4. * 3)
# # print(conv2d(KW, Kinv, padding=1))
# # print(conv2d(Kinv, KW, padding=1))
# # print(KW, Kinv, conv2d(KW, Kinv, padding=1), conv2d(Kinv, KW, padding=1))

# cuda = True if torch.cuda.is_available() else False
# if cuda:
#     KW = KW.to('cuda')
#     Kinv = Kinv.to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    do_tensorboard = True
except:  # ImportError:
    do_tensorboard = False
    print("Impossible de charger Tensorboard.")

def learn(opt):
    path_data = os.path.join("./runs", opt.runs_path)
    if not os.path.isdir(path_data):
        os.makedirs(path_data, exist_ok=True)
        do_learn(opt)

def do_learn(opt):
    print('Starting ', opt.runs_path)
    path_data = os.path.join("./runs", opt.runs_path) # + '_' + tag)
    # ----------
    #  Tensorboard
    # ----------
    if do_tensorboard:
        # stats are stored in "runs", within subfolder opt.runs_path.
        writer = SummaryWriter(log_dir=path_data)

    # Create a time tag
    try:
        tag = datetime.datetime.now().isoformat(sep='_', timespec='seconds')
    except TypeError:
        # Python 3.5 and below
        # 'timespec' is an invalid keyword argument for this function
        tag = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_')
    tag = tag.replace(':', '-')


    # Configure data loader
    dataloader = load_data(opt.datapath, opt.img_size, opt.batch_size,
                           rand_hflip=opt.rand_hflip, rand_affine=opt.rand_affine)

    # Loss functions
    adversarial_loss = torch.nn.BCEWithLogitsLoss()  # eq. 8 in https://arxiv.org/pdf/1701.00160.pdf
    if opt.do_SSIM:
        E_loss = NMSSSIM(window_size=opt.window_size, val_range=1., size_average=True, channel=3, normalize=True)
    else:
        E_loss = torch.nn.MSELoss(reduction='sum')
    MSE_loss = torch.nn.MSELoss(reduction='sum')
    sigmoid = nn.Sigmoid()

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)
    if opt.latent_threshold>0.:
        hs = torch.nn.Hardshrink(lambd=opt.latent_threshold)

    if opt.verbose:
        print_network(generator)
        print_network(discriminator)
        print_network(encoder)

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        #print("Nombre de GPU : ",torch.cuda.device_count())
        print("Running on GPU : ", torch.cuda.get_device_name())
        # if torch.cuda.device_count() > opt.GPU:
        #     torch.cuda.set_device(opt.GPU)
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        encoder.cuda()
        MSE_loss.cuda()
        E_loss.cuda()

    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    # Initialize weights
    if opt.init_weight:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        encoder.apply(weights_init_normal)

    # Optimizers
    # https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lrG, momentum=1-opt.beta1, alpha=opt.beta2)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lrD, momentum=1-opt.beta1, alpha=opt.beta2)
    if opt.do_joint:
        import itertools
        optimizer_E = torch.optim.RMSprop(itertools.chain(
            encoder.parameters(), generator.parameters()), lr=opt.lrE, momentum=1-opt.beta1, alpha=opt.beta2)
    else:
        optimizer_E = torch.optim.RMSprop(encoder.parameters(), lr=opt.lrE, momentum=1-opt.beta1, alpha=opt.beta2)

    # ----------
    #  Load models
    # ----------
    # start_epoch = 1
    # if opt.load_model == True:
    #     start_epoch = load_models(discriminator, optimizer_D, generator,
    #                               optimizer_G, opt.n_epochs, opt.model_save_path, encoder, optimizer_E)


    # ----------
    #  Training
    # ----------

    nb_batch = len(dataloader)
    # nb_epochs = 1 + opt.n_epochs - start_epoch

    hist = init_hist(opt.n_epochs, nb_batch)

    # save_dot = 1 # Nombre d'epochs avant de sauvegarder un point des courbes
    # batch_on_save_dot = save_dot*len(dataloader)

    # Vecteur z fixe pour faire les samples
    fixed_noise = Variable(Tensor(np.random.normal(
        0, 1, (opt.N_samples, opt.latent_dim))), requires_grad=False)
    real_imgs_samples = None

    def gen_z(threshold=opt.latent_threshold):
        z = np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))
        if threshold > 0:
            z[np.abs(z)<threshold] = 0.
        z = Variable(Tensor(z), requires_grad=False)
        return z


    zero_target = Variable(Tensor(torch.zeros(opt.batch_size, opt.channels,
                                              opt.img_size, opt.img_size)), requires_grad=False)
    z_zeros = Variable(Tensor(opt.batch_size, opt.latent_dim).fill_(0), requires_grad=False)
    z_ones = Variable(Tensor(opt.batch_size, opt.latent_dim).fill_(1), requires_grad=False)
    valid = Variable(Tensor(opt.batch_size, 1).fill_(1), requires_grad=False)
    fake = Variable(Tensor(opt.batch_size, 1).fill_(0), requires_grad=False)

    t_total = time.time()
    for j, epoch in enumerate(range(1, opt.n_epochs + 1)):
        t_epoch = time.time()
        for i, (imgs, _) in enumerate(dataloader):
            t_batch = time.time()

            # ---------------------
            #  Train Encoder
            # ---------------------
            for p in generator.parameters():
                p.requires_grad = opt.do_joint
            for p in encoder.parameters():
                p.requires_grad = True
            for p in discriminator.parameters():
                p.requires_grad = False  # to avoid computation

            real_imgs = Variable(imgs.type(Tensor), requires_grad=False)

            # init samples used to visualize performance of the AE
            if real_imgs_samples is None:
                real_imgs_samples = real_imgs[:opt.N_samples]

            # add noise here to real_imgs
            real_imgs_ = real_imgs * 1.
            if opt.E_noise > 0:
                noise = Variable(Tensor(np.random.normal(0, opt.E_noise, real_imgs.shape)), requires_grad=False)
                real_imgs_ = real_imgs_ + noise

            z_imgs = encoder(real_imgs_)
            if opt.latent_threshold>0:
                z_imgs = hs(z_imgs)

            decoded_imgs = generator(z_imgs)
            # decoded_imgs_samples = decoded_imgs[:opt.N_samples]

            optimizer_E.zero_grad()

            # Loss measures Encoder's ability to generate vectors suitable with the generator
            energy = 1. # E_loss(real_imgs, zero_target)  # normalize on the energy of imgs
            if opt.do_joint:
                e_loss = E_loss(real_imgs, decoded_imgs) / energy
            else:
                e_loss = E_loss(real_imgs, decoded_imgs.detach()) / energy

            if opt.lambdaE > 0:
                # We do not do a VAE, still we wish to make sure the z_imgs get closer to a gaussian
                # https://github.com/pytorch/examples/blob/master/vae/main.py#L72
                # # see Appendix B from VAE paper:
                # # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                # # https://arxiv.org/abs/1312.6114
                # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # add a loss for the distance between of z values
                e_loss += opt.lambdaE * MSE_loss(z_imgs, z_zeros)/opt.batch_size/opt.latent_dim
                e_loss += opt.lambdaE * \
                    MSE_loss(z_imgs.pow(2), z_ones).pow(.5)/opt.batch_size/opt.latent_dim

            # Backward
            e_loss.backward()
            optimizer_E.step()

            if opt.lrD > 0:

                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Discriminator Requires grad, Encoder + Generator requires_grad = False
                for p in discriminator.parameters():
                    p.requires_grad = True
                for p in generator.parameters():
                    p.requires_grad = False  # to avoid computation
                for p in encoder.parameters():
                    p.requires_grad = False  # to avoid computation

            # Configure input
            real_imgs = Variable(imgs.type(Tensor), requires_grad=False)

            real_imgs_ = real_imgs * 1.
            if opt.D_noise > 0:
                # real_imgs_ += opt.D_noise * Variable(torch.randn(real_imgs.shape))
                noise = Variable(Tensor(np.random.normal(0, opt.D_noise, real_imgs.shape)), requires_grad=False)
                real_imgs_ = real_imgs_ + noise

            # Real batch
            # Discriminator decision (in logit units)
            d_x = discriminator(real_imgs_)

            # Adversarial ground truths
            valid_smooth = Variable(Tensor(imgs.shape[0], 1).fill_(
                float(np.random.uniform(opt.valid_smooth, 1.0, 1))), requires_grad=False)

            if opt.lrD > 0:

                # ---------------------
                #  Train Discriminator
                # ---------------------
                if opt.GAN_loss == 'wasserstein':
                    # weight clipping
                    for p in discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)


                optimizer_D.zero_grad()


                # Measure discriminator's ability to classify real from generated samples
                if opt.GAN_loss == 'ian':
                    # eq. 14 in https://arxiv.org/pdf/1701.00160.pdf
                    real_loss = - torch.sum(1 / (1. - 1/sigmoid(d_x)))
                elif opt.GAN_loss == 'wasserstein':
                    real_loss = torch.mean(torch.abs(valid_smooth - sigmoid(d_x)))
                elif opt.GAN_loss == 'alternative':
                    # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                    real_loss = - torch.sum(torch.log(sigmoid(d_x)))
                elif opt.GAN_loss == 'alternativ2':
                    # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                    real_loss = - torch.sum(torch.log(sigmoid(d_x) / (1. - sigmoid(d_x))))
                elif opt.GAN_loss == 'original':
                    real_loss = adversarial_loss(d_x, valid_smooth)
                else:
                    print ('GAN_loss not defined', opt.GAN_loss)

                # Backward
                real_loss.backward()

            # Generate a batch of fake images
            z = gen_z()
            if opt.latent_threshold>0:
                z_imgs = hs(z_imgs)
            gen_imgs = generator(z)
            # Discriminator decision for fake data
            gen_imgs_ = gen_imgs * 1.
            # if opt.D_noise > 0:
            #     gen_imgs_ += opt.D_noise * Variable(torch.randn(gen_imgs.shape))
            if opt.D_noise > 0:
                # real_imgs_ += opt.D_noise * Variable(torch.randn(real_imgs.shape))
                noise = Variable(Tensor(np.random.normal(0, opt.D_noise, gen_imgs.shape)), requires_grad=False)
                gen_imgs_ = gen_imgs_ + noise

            d_fake = discriminator(gen_imgs_.detach())

            if opt.lrD > 0:

                # Measure discriminator's ability to classify real from generated samples
                if opt.GAN_loss == 'ian':
                    # eq. 14 in https://arxiv.org/pdf/1701.00160.pdf
                    fake_loss = - torch.sum(1 / (1. - 1/(1-sigmoid(d_fake))))
                elif opt.GAN_loss == 'wasserstein':
                    fake_loss = torch.mean(sigmoid(d_fake))
                elif opt.GAN_loss == 'alternative':
                    # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                    fake_loss = - torch.sum(torch.log(1-sigmoid(d_fake)))
                elif opt.GAN_loss == 'alternativ2':
                    # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                    fake_loss = torch.sum(torch.log(sigmoid(d_fake) / (1. - sigmoid(d_fake))))
                elif opt.GAN_loss == 'original':
                    fake_loss = adversarial_loss(d_fake, fake)
                else:
                    print ('GAN_loss not defined', opt.GAN_loss)

                # Backward
                fake_loss.backward()

                d_loss = real_loss + fake_loss

                optimizer_D.step()

            if opt.lrG > 0:
                # -----------------
                #  Train Generator
                # -----------------
                # TODO : optimiser la distance z - E(G(z))
                for p in generator.parameters():
                    p.requires_grad = True
                for p in discriminator.parameters():
                    p.requires_grad = False  # to avoid computation
                for p in encoder.parameters():
                    p.requires_grad = False  # to avoid computation

            # Generate a batch of fake images
            z = gen_z()
            gen_imgs = generator(z)
            # New discriminator decision (since we just updated D)
            gen_imgs_ = gen_imgs * 1.
            if opt.G_noise > 0:
                # gen_imgs_ += opt.G_noise * Variable(torch.randn(gen_imgs.shape))
                noise = Variable(Tensor(np.random.normal(0, opt.G_noise, gen_imgs.shape)), requires_grad=False)
                gen_imgs_ = gen_imgs_ + noise
            d_g_z = discriminator(gen_imgs_)

            if opt.lrG > 0:
                optimizer_G.zero_grad()

                # Loss measures generator's ability to fool the discriminator
                if opt.GAN_loss == 'ian':
                    # eq. 14 in https://arxiv.org/pdf/1701.00160.pdf
                    g_loss = - torch.sum(1 / (1. - 1/sigmoid(d_g_z)))
                elif opt.GAN_loss == 'wasserstein':
                    g_loss = torch.mean(torch.abs(valid - sigmoid(d_g_z)))
                elif opt.GAN_loss == 'alternative':
                    # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                    #g_loss = - adversarial_loss(1-d_g_z, valid)
                    g_loss = - torch.sum(torch.log(sigmoid(d_g_z)))
                elif opt.GAN_loss == 'alternativ2':
                    # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                    g_loss = - torch.sum(torch.log(sigmoid(d_g_z) / (1. - sigmoid(d_g_z))))
                elif opt.GAN_loss == 'original':
                    g_loss = adversarial_loss(d_g_z, valid)
                else:
                    print ('GAN_loss not defined', opt.GAN_loss)

                # Backward
                g_loss.backward()
                optimizer_G.step()


            # -----------------
            #  Recording stats
            # -----------------
            if opt.lrG > 0:
                # Compensation pour le BCElogits
                # d_fake = sigmoid(d_fake)
                d_x = sigmoid(d_x)
                d_g_z = sigmoid(d_g_z)
                print(
                    "%s [Epoch %d/%d] [Batch %d/%d] [E loss: %f] [D loss: %f] [G loss: %f] [D score %f] [G score %f] [Time: %fs]"
                    % (opt.runs_path, epoch, opt.n_epochs, i+1, len(dataloader), e_loss.item(), d_loss.item(), g_loss.item(), torch.mean(d_x), torch.mean(d_g_z), time.time()-t_batch)
                )
                # Save Losses and scores for Tensorboard
                save_hist_batch(hist, i, j, g_loss, d_loss, e_loss, d_x, d_g_z)
            else:
                print(
                    "%s [Epoch %d/%d] [Batch %d/%d] [E loss: %f] [Time: %fs]"
                    % (opt.runs_path, epoch, opt.n_epochs, i+1, len(dataloader), e_loss.item(), time.time()-t_batch)
                )

            if do_tensorboard:
                # Tensorboard save
                iteration = i + nb_batch * j
                writer.add_scalar('loss/E', e_loss.item(), global_step=iteration)
                writer.add_histogram('coeffs/z', z, global_step=iteration)
                try:
                    writer.add_histogram('coeffs/E_x', z_imgs, global_step=iteration)
                except:
                    pass
                writer.add_histogram('image/x', real_imgs, global_step=iteration)
                try:
                    writer.add_histogram('image/E_G_x', decoded_imgs, global_step=iteration)
                except:
                    pass
                try:
                    writer.add_histogram('image/G_z', gen_imgs, global_step=iteration)
                except:
                    pass
                if opt.lrG > 0:
                    writer.add_scalar('loss/G', g_loss.item(), global_step=iteration)
                    # writer.add_scalar('score/D_fake', hist["d_fake_mean"][i], global_step=iteration)
                    writer.add_scalar('score/D_g_z', hist["d_g_z_mean"][i], global_step=iteration)
                    try:
                        writer.add_histogram('D_G_z', d_g_z, global_step=iteration,
                                             bins=np.linspace(0, 1, 20))
                    except:
                        pass
                if opt.lrD > 0:
                    writer.add_scalar('loss/D', d_loss.item(), global_step=iteration)

                    writer.add_scalar('score/D_x', hist["d_x_mean"][i], global_step=iteration)

                    # writer.add_scalar('d_x_cv', hist["d_x_cv"][i], global_step=iteration)
                    # writer.add_scalar('d_g_z_cv', hist["d_g_z_cv"][i], global_step=iteration)
                    try:
                        writer.add_histogram('D_x', d_x, global_step=iteration,
                                         bins=np.linspace(0, 1, 20))
                    except:
                        pass


        if do_tensorboard:
            # writer.add_scalar('D_x/max', hist["D_x_max"][j], global_step=epoch)
            # writer.add_scalar('D_x/min', hist["D_x_min"][j], global_step=epoch)
            # writer.add_scalar('D_G_z/min', hist["D_G_z_min"][j], global_step=epoch)
            # writer.add_scalar('D_G_z/max', hist["D_G_z_max"][j], global_step=epoch)

            # Save samples
            if epoch % opt.sample_interval == 0:
                tensorboard_sampling(fixed_noise, generator, writer, epoch)
                tensorboard_AE_comparator(real_imgs_samples, generator, encoder, writer, epoch) # TODO use decoded_imgs_samples

        if epoch % opt.sample_interval == 0 :
            sampling(fixed_noise, generator, path_data, epoch, tag)
            # do_plot(hist, start_epoch, epoch)

        # # Save models
        # if epoch % opt.model_save_interval == 0 :
        #     num = str(int(epoch / opt.model_save_interval))
        #     save_model(discriminator, optimizer_D, epoch, opt.model_save_path + "/" + num + "_D.pt")
        #     save_model(generator, optimizer_G, epoch, opt.model_save_path + "/" + num + "_G.pt")
        #     save_model(encoder, optimizer_E, epoch, opt.model_save_path + "/" + num + "_E.pt")

        print("[Epoch Time: ", time.time() - t_epoch, "s]")

    t_final = time.gmtime(time.time() - t_total)
    print("[Total Time: ", t_final.tm_mday - 1, "j:",
          time.strftime("%Hh:%Mm:%Ss", t_final), "]", sep='')

    # Save model for futur training
    # if opt.model_save_interval < opt.n_epochs + 1:
    #     save_model(discriminator, optimizer_D, epoch, opt.model_save_path + "/last_D.pt")
    #     save_model(generator, optimizer_G, epoch, opt.model_save_path + "/last_G.pt")
    #     save_model(encoder, optimizer_E, epoch, opt.model_save_path + "/last_E.pt")

    if do_tensorboard:
        writer.close()
