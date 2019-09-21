import os
import numpy as np

import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image
try:
    from torch.utils.tensorboard import SummaryWriter
    do_tensorboard = True
except: # ImportError:
    do_tensorboard = False
    print("Impossible de charger Tensorboard.")

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import sys

import matplotlib.pyplot as plt
import time
import datetime

from .utils import *
from .plot import *
from .models import *

def learn(opt):
    print('Starting ', opt.runs_path)

    # Create a time tag
    try:
        tag = datetime.datetime.now().isoformat(sep='_', timespec='seconds')
    except TypeError:
        # Python 3.5 and below
        # 'timespec' is an invalid keyword argument for this function
        tag = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_')
    tag = tag.replace(':','-')

    cuda = True if torch.cuda.is_available() else False

    # Loss function
    adversarial_loss = torch.nn.BCEWithLogitsLoss() # eq. 8 in https://arxiv.org/pdf/1701.00160.pdf
    MSE_loss = torch.nn.MSELoss()
    sigmoid = nn.Sigmoid()

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    if opt.verbose:
        print_network(generator)
        print_network(discriminator)
        print_network(encoder)

    if cuda:
        #print("Nombre de GPU : ",torch.cuda.device_count())
        if torch.cuda.device_count() > opt.GPU:
            torch.cuda.set_device(opt.GPU)

        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        encoder.cuda()
        MSE_loss.cuda()

    # Initialize weights
    if opt.init_weight:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        encoder.apply(weights_init_normal)

    # Configure data loader
    dataloader = load_data(opt.datapath, opt.img_size, opt.batch_size, rand_hflip=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lrG, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lrD, betas=(opt.b1, opt.b2))
    optimizer_E = torch.optim.Adam(itertools.chain(encoder.parameters(), generator.parameters()), lr=opt.lrE, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Load models
    # ----------
    start_epoch = 1
    if opt.load_model == True:
        start_epoch = load_models(discriminator, optimizer_D, generator, optimizer_G, opt.n_epochs, opt.model_save_path, encoder, optimizer_E)

    path_data = os.path.join("./runs", opt.runs_path + '_' + tag)
    # ----------
    #  Tensorboard
    # ----------
    if do_tensorboard:
        # stats are stored in "runs", within subfolder opt.runs_path.
        os.makedirs(path_data, exist_ok=True)
        writer = SummaryWriter(log_dir=path_data)

    # ----------
    #  Training
    # ----------

    nb_batch = len(dataloader)
    nb_epochs = 1 + opt.n_epochs - start_epoch

    hist = init_hist(nb_epochs, nb_batch)

    # save_dot = 1 # Nombre d'epochs avant de sauvegarder un point des courbes
    # batch_on_save_dot = save_dot*len(dataloader)

    # Vecteur z fixe pour faire les samples
    fixed_noise = Variable(Tensor(np.random.normal(0, 1, (opt.N_samples, opt.latent_dim))))

    t_total = time.time()
    for j, epoch in enumerate(range(start_epoch, opt.n_epochs + 1)):
        t_epoch = time.time()
        for i, (imgs, _) in enumerate(dataloader):
            t_batch = time.time()
            # ---------------------
            #  Train Encoder
            # ---------------------

            real_imgs = Variable(imgs.type(Tensor))

            optimizer_E.zero_grad()
            z_imgs = encoder(real_imgs)
            decoded_imgs = generator(z_imgs)

            # Loss measures Encoder's ability to generate vectors suitable with the generator
            # DONE add a loss for the distance between of z values
            z_zeros = Variable(Tensor(z_imgs.size(0), z_imgs.size(1)).fill_(0), requires_grad=False)
            z_ones = Variable(Tensor(z_imgs.size(0), z_imgs.size(1)).fill_(1), requires_grad=False)
            e_loss = MSE_loss(real_imgs, decoded_imgs)
            e_loss += opt.lambdaE*MSE_loss(z_imgs, z_zeros)
            e_loss += opt.lambdaE*MSE_loss(z_imgs.pow(2), z_ones).pow(.5)

            # Backward
            e_loss.backward()

            optimizer_E.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Adversarial ground truths
            valid_smooth = Variable(Tensor(imgs.shape[0], 1).fill_(float(np.random.uniform(opt.valid_smooth, 1.0, 1))), requires_grad=False)
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            # Generate a batch of images
            z = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
            z = Variable(Tensor(z))
            gen_imgs = generator(z)

            optimizer_D.zero_grad()

            # Real batch
            # Discriminator decision (in logit units)
            d_x = discriminator(real_imgs)
            # Measure discriminator's ability to classify real from generated samples
            if opt.G_loss == 'wasserstein':
                real_loss = torch.mean(torch.abs(valid_smooth - sigmoid(d_x)))
            else:
                real_loss = adversarial_loss(d_x, valid_smooth)
            # Backward
            real_loss.backward()

            # Fake batch
            # Discriminator decision
            d_g_z = discriminator(gen_imgs.detach())
            # Measure discriminator's ability to classify real from generated samples
            if opt.G_loss == 'wasserstein':
                fake_loss = torch.mean(sigmoid(d_g_z))
            else:
                fake_loss = adversarial_loss(d_g_z, fake)
            # Backward
            fake_loss.backward()

            d_loss = real_loss + fake_loss

            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # New discriminator descision, Since we just updated D
            d_g_z = discriminator(gen_imgs)
            # Loss measures generator's ability to fool the discriminator
            if opt.G_loss == 'ian':
                # eq. 14 in https://arxiv.org/pdf/1701.00160.pdf
                g_loss = - torch.sum(1 / (1. - 1/sigmoid(d_g_z)))
            elif opt.G_loss == 'wasserstein':
                g_loss = torch.mean(torch.abs(valid - sigmoid(d_g_z)))
            elif opt.G_loss == 'alternative':
                # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                #g_loss = - adversarial_loss(1-d_g_z, valid)
                g_loss = - torch.sum(torch.log(sigmoid(d_g_z)))
            elif opt.G_loss == 'alternativ2':
                # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                g_loss = - torch.sum(torch.log(sigmoid(d_g_z) / (1. - sigmoid(d_g_z))))
            else:
                g_loss = adversarial_loss(d_g_z, valid)
            # Backward
            g_loss.backward()

            optimizer_G.step()

            # Compensation pour le BCElogits
            d_x = sigmoid(d_x)
            d_g_z = sigmoid(d_g_z)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [E loss: %f] [D loss: %f] [G loss: %f] [D score %f] [G score %f] [Time: %fs]"
                % (epoch, opt.n_epochs, i+1, len(dataloader), e_loss.item(), d_loss.item(), g_loss.item(), torch.mean(d_x), torch.mean(d_g_z), time.time()-t_batch)
            )

            # Save Losses and scores for Tensorboard
            save_hist_batch(hist, i, j, g_loss, d_loss, d_x, d_g_z)

            if do_tensorboard:
                # Tensorboard save
                iteration = i + nb_batch * j
                writer.add_scalar('e_loss', e_loss.item(), global_step=iteration)
                writer.add_scalar('g_loss', g_loss.item(), global_step=iteration)
                writer.add_scalar('d_loss', d_loss.item(), global_step=iteration)

                writer.add_scalar('d_x_mean', hist["d_x_mean"][i], global_step=iteration)
                writer.add_scalar('d_g_z_mean', hist["d_g_z_mean"][i], global_step=iteration)

                writer.add_scalar('d_x_cv', hist["d_x_cv"][i], global_step=iteration)
                writer.add_scalar('d_g_z_cv', hist["d_g_z_cv"][i], global_step=iteration)

                writer.add_histogram('D(x)', d_x, global_step=iteration)
                writer.add_histogram('D(G(z))', d_g_z, global_step=iteration)

        if do_tensorboard:
            writer.add_scalar('D_x_max', hist["D_x_max"][j], global_step=epoch)
            writer.add_scalar('D_x_min', hist["D_x_min"][j], global_step=epoch)
            writer.add_scalar('D_G_z_min', hist["D_G_z_min"][j], global_step=epoch)
            writer.add_scalar('D_G_z_max', hist["D_G_z_max"][j], global_step=epoch)

            # Save samples
            if epoch % opt.sample_interval == 0:
                tensorboard_sampling(fixed_noise, generator, writer, epoch)
                tensorboard_AE_comparator(real_imgs[:opt.N_samples], generator, encoder, writer, epoch)

        if epoch % opt.sample_interval == 0:
            sampling(fixed_noise, generator, path_data, epoch, tag)
            do_plot(hist, start_epoch, epoch)

        # Save models
        if epoch % opt.model_save_interval == 0:
            num = str(int(epoch / opt.model_save_interval))
            save_model(discriminator, optimizer_D, epoch, opt.model_save_path + "/" + num + "_D.pt")
            save_model(generator, optimizer_G, epoch, opt.model_save_path + "/" + num + "_G.pt")
            save_model(encoder, optimizer_E, epoch, opt.model_save_path + "/" + num + "_E.pt")

        print("[Epoch Time: ", time.time() - t_epoch, "s]")

    durer = time.gmtime(time.time() - t_total)
    print("[Total Time: ", durer.tm_mday - 1, "j:", time.strftime("%Hh:%Mm:%Ss", durer), "]", sep='')

    # Save model for futur training
    if opt.model_save_interval < opt.n_epochs + 1:
        save_model(discriminator, optimizer_D, epoch, opt.model_save_path + "/last_D.pt")
        save_model(generator, optimizer_G, epoch, opt.model_save_path + "/last_G.pt")
        save_model(encoder, optimizer_E, epoch, opt.model_save_path + "/last_E.pt")

    if do_tensorboard: writer.close()
