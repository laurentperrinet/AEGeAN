import os
import time
import numpy as np

import torch
torch.autograd.set_detect_anomaly(True)
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

# from .init import init
from .utils import print_network, sampling
from .utils import init_hist, load_data, weights_init_normal, save_hist_batch
from .models import Generator, Discriminator, Encoder

try:
    # to use with `$ tensorboard --logdir runs`
    from torch.utils.tensorboard import SummaryWriter
    do_tensorboard = True
except:  # ImportError:
    do_tensorboard = False
    print("Failed loading Tensorboard.")

def learn(opt, run_dir="./runs"):
    os.makedirs(run_dir, exist_ok=True)
    path_data = os.path.join(run_dir, opt.run_path)
    if not os.path.isdir(path_data):
        os.makedirs(path_data, exist_ok=True)
        do_learn(opt)

def do_learn(opt, run_dir="./runs"):
    print('Starting ', opt.run_path)
    path_data = os.path.join(run_dir, opt.run_path)
    # ----------
    #  Tensorboard
    # ----------
    if do_tensorboard:
        # stats are stored in "runs", within subfolder opt.run_path.
        writer = SummaryWriter(log_dir=path_data)

    # Create a time tag
    import datetime
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


    if opt.do_SSIM:
        # from pytorch_msssim import NMSSSIM as neg_SSIM
        from pytorch_msssim import NSSIM as neg_SSIM
        E_loss = neg_SSIM(window_size=opt.window_size, val_range=1., size_average=True, channel=3, normalize=True)
    else:
        E_loss = torch.nn.MSELoss(reduction='sum')

    sigmoid = torch.nn.Sigmoid()

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    if opt.verbose:
        print_network(generator)
        print_network(discriminator)
        print_network(encoder)

    eye = 1 - torch.eye(opt.batch_size)
    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        #print("Nombre de GPU : ",torch.cuda.device_count())
        print("Running on GPU : ", torch.cuda.get_device_name())
        # if torch.cuda.device_count() > opt.GPU:
        #     torch.cuda.set_device(opt.GPU)
        generator.cuda()
        discriminator.cuda()
        # adversarial_loss.cuda()
        encoder.cuda()
        # MSE_loss.cuda()
        E_loss.cuda()
        eye = eye.cuda()

        Tensor = torch.cuda.FloatTensor
    else:
        print("Running on CPU ")
        Tensor = torch.FloatTensor

    # Initialize weights
    if opt.init_weight:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        encoder.apply(weights_init_normal)

    # Optimizers
    if opt.optimizer == 'rmsprop':
        # https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop
        opts = dict(momentum=1-opt.beta1, alpha=opt.beta2)
        optimizer = torch.optim.RMSprop
    elif opt.optimizer == 'adam':
        # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
        opts = dict(betas=(opt.beta1, opt.beta2))
        optimizer = torch.optim.Adam
    elif opt.optimizer == 'sgd':
        opts = dict(momentum=1-opt.beta1, nesterov=True, weight_decay=1-opt.beta2)
        optimizer = torch.optim.SGD
    else:
        raise('wrong optimizer')

    optimizer_G = optimizer(generator.parameters(), lr=opt.lrG, **opts)
    optimizer_D = optimizer(discriminator.parameters(), lr=opt.lrD, **opts)
    if opt.do_joint:
        import itertools
        optimizer_E = optimizer(itertools.chain(encoder.parameters(), generator.parameters()), lr=opt.lrE, **opts)
    else:
        optimizer_E = optimizer(encoder.parameters(), lr=opt.lrE, **opts)

    # TODO parameterize scheduler !
    # gamma = .1 ** (1 / opt.n_epochs)
    # schedulers = []
    # for optimizer in [optimizer_G, optimizer_D, optimizer_E]:
    #     schedulers.append(torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma))

    # ----------
    #  Training
    # ----------

    nb_batch = len(dataloader)

    stat_record = init_hist(opt.n_epochs, nb_batch)

    # https://github.com/soumith/dcgan.torch/issues/14  dribnet commented on 21 Mar 2016
    # https://arxiv.org/abs/1609.04468
    def slerp(val, low, high):
        corr = np.diag((low/np.linalg.norm(low)) @ (high/np.linalg.norm(high)).T)
        omega = np.arccos(np.clip(corr, -1, 1))[:, None]
        so = np.sin(omega)
        out =  np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
        # L'Hopital's rule/LERP
        out[so[:, 0] == 0, :] = (1.0-val) * low[so[:, 0] == 0, :] + val * high[so[:, 0] == 0, :]
        return out

    def norm2(z):
        """
        L2-norm of a tensor.

        outputs a scalar
        """
        # return torch.mean(z.pow(2)).pow(.5)
        return (z**2).sum().sqrt()

    def gen_z(imgs=None, rho=.25, do_slerp=opt.do_slerp):
        """
        Generate noise in the feature space.

        outputs a vector
        """
        if not imgs is None:
            z_imgs = encoder(imgs).cpu().numpy()
            if do_slerp:
                z_shuffle = z_imgs.copy()
                z_shuffle = z_shuffle[torch.randperm(opt.batch_size), :]
                z = slerp(rho, z_imgs, z_shuffle)
            else:
                z /= norm2(z)
                z_imgs /= norm2(z_imgs)
                z = (1-rho) * z_imgs + rho * z
                z /= norm2(z)
        else:
            z = np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))
        # convert to tensor
        return Variable(Tensor(z), requires_grad=False)

    def gen_noise(imgs):
        """
        Generate noise in the image space

        outputs an image
        """
        v_noise = np.random.normal(0, 1, imgs.shape) # one random image
        # one contrast value per image
        v_noise *= np.abs(np.random.normal(0, 1, (imgs.shape[0], opt.channels, 1, 1)))
        # convert to tensor
        v_noise =  Variable(Tensor(v_noise), requires_grad=False)
        return v_noise

    # Vecteur z fixe pour faire les samples
    fixed_noise = gen_z()
    real_imgs_samples = None

    # z_zeros = Variable(Tensor(opt.batch_size, opt.latent_dim).fill_(0), requires_grad=False)
    # z_ones = Variable(Tensor(opt.batch_size, opt.latent_dim).fill_(1), requires_grad=False)
    # Adversarial ground truths
    # valid = Variable(Tensor(opt.batch_size, 1).fill_(1), requires_grad=False)
    # fake = Variable(Tensor(opt.batch_size, 1).fill_(0), requires_grad=False)

    t_total = time.time()
    for i_epoch, epoch in enumerate(range(1, opt.n_epochs + 1)):
        t_epoch = time.time()
        for iteration, (imgs, _) in enumerate(dataloader):
            t_batch = time.time()

            # ---------------------
            #  Train Encoder
            # ---------------------
            for p in generator.parameters():
                p.requires_grad = opt.do_joint
            for p in encoder.parameters():
                p.requires_grad = True
            # the following is not necessary as we do not use D here and only optimize ||G(E(x)) - x ||^2
            for p in discriminator.parameters():
                p.requires_grad = False  # to avoid learning D when learning E

            real_imgs = Variable(imgs.type(Tensor), requires_grad=False)

            # init samples used to visualize performance of the AE
            if real_imgs_samples is None:
                real_imgs_samples = real_imgs[:opt.N_samples]

            # add noise here to real_imgs
            real_imgs_ = real_imgs * 1.
            if opt.E_noise > 0: real_imgs_ += opt.E_noise * gen_noise(real_imgs)

            z_imgs = encoder(real_imgs_)
            decoded_imgs = generator(z_imgs)

            optimizer_E.zero_grad()

            # Loss measures Encoder's ability to generate vectors suitable with the generator
            e_loss = E_loss(real_imgs, decoded_imgs)
            # energy = 1. # E_loss(real_imgs, zero_target)  # normalize on the energy of imgs
            # if opt.do_joint:
            #     e_loss = E_loss(real_imgs, decoded_imgs) / energy
            # else:
            #     e_loss = E_loss(real_imgs, decoded_imgs.detach()) / energy

            if opt.lambdaE > 0:
                # We wish to make sure the intermediate vector z_imgs get closer to a iid normal (centered gausian of variance 1)
                e_loss += opt.lambdaE * (torch.sum(z_imgs)/opt.batch_size/opt.latent_dim).pow(2)
                e_loss += opt.lambdaE * (torch.sum(z_imgs.pow(2))/opt.batch_size/opt.latent_dim-1).pow(2).pow(.5)

            # Backward
            e_loss.backward()
            optimizer_E.step()

            valid_smooth = np.random.uniform(opt.valid_smooth, 1.0-(1-opt.valid_smooth)/2, (opt.batch_size, 1))
            valid_smooth = Variable(Tensor(valid_smooth), requires_grad=False)
            fake_smooth = np.random.uniform((1-opt.valid_smooth)/2, 1-opt.valid_smooth, (opt.batch_size, 1))
            fake_smooth = Variable(Tensor(fake_smooth), requires_grad=False)

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
            if opt.D_noise > 0: real_imgs_ += opt.D_noise * gen_noise(real_imgs)
            if opt.do_insight:
                # the discriminator can not access the images directly but only
                # what is visible through the auto-encoder
                real_imgs_ = generator(encoder(real_imgs_))

            # Discriminator decision (in logit units)
            # TODO : group images by sub-batches and train to discriminate from all together
            # should allow to avoid mode collapse
            logit_d_x = discriminator(real_imgs_)

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
                real_loss = - torch.sum(1 / (1. - 1/sigmoid(logit_d_x)))
            elif opt.GAN_loss == 'hinge':
                # TODO check if we use p or log p
                real_loss = nn.ReLU()(valid_smooth - sigmoid(logit_d_x)).mean()
            elif opt.GAN_loss == 'wasserstein':
                real_loss = torch.mean(torch.abs(valid_smooth - sigmoid(logit_d_x)))
            elif opt.GAN_loss == 'alternative':
                # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                real_loss = - torch.sum(torch.log(sigmoid(logit_d_x)))
            elif opt.GAN_loss == 'alternativ2':
                # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                real_loss = - torch.sum(torch.log(sigmoid(logit_d_x) / (1. - sigmoid(logit_d_x))))
            elif opt.GAN_loss == 'alternativ3':
                # to maximize D(x), we minimize  - sum(logit_d_x)
                real_loss = - torch.sum(logit_d_x)
            elif opt.GAN_loss == 'original':
                real_loss = F.binary_cross_entropy(sigmoid(logit_d_x), valid_smooth)
            else: print ('GAN_loss not defined', opt.GAN_loss)

            # Backward
            real_loss.backward()

            # Generate a batch of fake images and learn the discriminator to treat them as such
            z = gen_z(imgs=real_imgs_)
            gen_imgs = generator(z)
            if opt.D_noise > 0: gen_imgs += opt.D_noise * gen_noise(real_imgs)

            # Discriminator decision for fake data
            logit_d_fake = discriminator(gen_imgs.detach())
            # Measure discriminator's ability to classify real from generated samples
            if opt.GAN_loss == 'wasserstein':
                fake_loss = torch.mean(sigmoid(logit_d_fake))
            elif opt.GAN_loss == 'hinge':
                # TODO check if we use p or log p
                real_loss = nn.ReLU()(1.0 + sigmoid(logit_d_fake)).mean()
            elif opt.GAN_loss == 'alternative':
                # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                fake_loss = - torch.sum(torch.log(1-sigmoid(logit_d_fake)))
            elif opt.GAN_loss == 'alternativ2':
                # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                fake_loss = torch.sum(torch.log(sigmoid(logit_d_fake) / (1. - sigmoid(logit_d_fake))))
            elif opt.GAN_loss == 'alternativ3':
                # to minimize D(G(z)), we minimize sum(logit_d_fake)
                fake_loss = torch.sum(logit_d_fake)
            elif opt.GAN_loss in ['original', 'ian']:
                fake_loss = F.binary_cross_entropy(sigmoid(logit_d_fake), fake_smooth)
            else:
                print ('GAN_loss not defined', opt.GAN_loss)

            # Backward
            fake_loss.backward()
            # apply the gradients
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            for p in generator.parameters():
                p.requires_grad = True
            for p in discriminator.parameters():
                p.requires_grad = False  # to avoid computation
            for p in encoder.parameters():
                p.requires_grad = False  # to avoid computation

            # Generate a batch of fake images
            z = gen_z(imgs=real_imgs_)
            gen_imgs = generator(z)
            if opt.G_noise > 0: gen_imgs += opt.G_noise * gen_noise(real_imgs)

            # New discriminator decision (since we just updated D)
            logit_d_g_z = discriminator(gen_imgs)

            optimizer_G.zero_grad()
            # Loss functions
            # Loss measures generator's ability to fool the discriminator
            if opt.GAN_loss == 'ian':
                # eq. 14 in https://arxiv.org/pdf/1701.00160.pdf
                # https://en.wikipedia.org/wiki/Logit
                g_loss = - torch.sum(sigmoid(logit_d_g_z)/(1 - sigmoid(logit_d_g_z)))
            elif opt.GAN_loss == 'wasserstein' or  opt.GAN_loss == 'hinge':
                g_loss = torch.mean(torch.abs(valid_smooth - sigmoid(logit_d_g_z)))
            elif opt.GAN_loss == 'alternative':
                # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                g_loss = - torch.sum(torch.log(sigmoid(logit_d_g_z)))
            elif opt.GAN_loss == 'alternativ2':
                # https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/
                g_loss = - torch.sum(torch.log(sigmoid(logit_d_g_z) / (1. - sigmoid(logit_d_g_z))))
                # g_loss = torch.sum(torch.log(1./sigmoid(logit_d_g_z) - 1.))
            elif opt.GAN_loss == 'alternativ3':
                # to maximize D(G(z)), we minimize - sum(logit_d_g_z)
                g_loss = - torch.sum(logit_d_g_z)
            elif opt.GAN_loss == 'original':
                # https://pytorch.org/docs/stable/nn.html?highlight=bcewithlogitsloss#torch.nn.BCEWithLogitsLoss
                #adversarial_loss = torch.nn.BCEWithLogitsLoss()  # eq. 8 in https://arxiv.org/pdf/1701.00160.pdf
                #
                # https://medium.com/swlh/gan-to-generate-images-of-cars-5f706ca88da
                # adversarial_loss = torch.nn.BCE()  # eq. 8 in https://arxiv.org/pdf/1701.00160.pdf
                g_loss = F.binary_cross_entropy(sigmoid(logit_d_g_z), valid_smooth)
            else:
                print ('GAN_loss not defined', opt.GAN_loss)

            # penalize low variability in a batch, that is, mode collapse
            if opt.lambdaG > 0:
                e_g_z = encoder(gen_imgs) # get normal vectors
                Xcorr = torch.tensordot(e_g_z, torch.transpose(e_g_z, 0, 1), 1)/opt.latent_dim
                Xcorr *= eye
                g_loss += opt.lambdaG * torch.sum(Xcorr.pow(2)).pow(.5)

            # Backward
            g_loss.backward()
            # apply the gradients
            optimizer_G.step()


            # -----------------
            #  Recording stats
            # -----------------
            d_loss = real_loss + fake_loss

            # Compensation pour le BCElogits
            d_fake = sigmoid(logit_d_fake)
            d_x = sigmoid(logit_d_x)
            d_g_z = sigmoid(logit_d_g_z)
            print(
                "%s [Epoch %d/%d] [Batch %d/%d] [E loss: %f] [D loss: %f] [G loss: %f] [D(x) %f] [D(G(z)) %f] [D(G(z')) %f] [Time: %fs]"
                % (opt.run_path, epoch, opt.n_epochs, iteration+1, len(dataloader), e_loss.item(), d_loss.item(), g_loss.item(), torch.mean(d_x), torch.mean(d_fake), torch.mean(d_g_z), time.time()-t_batch)
            )
            # Save Losses and scores for Tensorboard
            save_hist_batch(stat_record, iteration, i_epoch, g_loss, d_loss, e_loss, d_x, d_g_z)

        if do_tensorboard:
            # Tensorboard save
            writer.add_scalar('loss/E', e_loss.item(), global_step=epoch)
            # writer.add_histogram('coeffs/z', z, global_step=epoch)
            try:
                writer.add_histogram('coeffs/E_x', z_imgs, global_step=epoch)
            except:
                pass
            # writer.add_histogram('image/x', real_imgs, global_step=epoch)
            # try:
            #     writer.add_histogram('image/E_G_x', decoded_imgs, global_step=epoch)
            # except:
            #     pass
            # try:
            #     writer.add_histogram('image/G_z', gen_imgs, global_step=epoch)
            # except:
            #     pass
            writer.add_scalar('loss/G', g_loss.item(), global_step=epoch)
            # writer.add_scalar('score/D_fake', hist["d_fake_mean"][i], global_step=epoch)
            # print(stat_record["d_g_z_mean"])
            writer.add_scalar('score/D_g_z', np.mean(stat_record["d_g_z_mean"]), global_step=epoch)
            writer.add_scalar('loss/D', d_loss.item(), global_step=epoch)

            writer.add_scalar('score/D_x', np.mean(stat_record["d_x_mean"]), global_step=epoch)

            # Save samples
            if epoch % opt.sample_interval == 0:
                """
                Use generator model and noise vector to generate images.
                Save them to tensorboard
                """
                generator.eval()
                gen_imgs = generator(fixed_noise)
                from torchvision.utils import make_grid
                grid = make_grid(gen_imgs, normalize=True, nrow=16, range=(0, 1))
                writer.add_image('Generated images', grid, epoch)
                generator.train()

                """
                Use auto-encoder model and original images to generate images.
                Save them to tensorboard

                """
                # grid_imgs = make_grid(real_imgs_samples, normalize=True, nrow=8, range=(0, 1))
                # writer.add_image('Images/original', grid_imgs, epoch)

                generator.eval()
                encoder.eval()
                enc_imgs = encoder(real_imgs_samples)
                dec_imgs = generator(enc_imgs)
                grid_dec = make_grid(dec_imgs, normalize=True, nrow=16, range=(0, 1))
                # writer.add_image('Images/auto-encoded', grid_dec, epoch)
                writer.add_image('Auto-encoded', grid_dec, epoch)
                generator.train()
                encoder.train()
                # writer.add_graph(encoder, real_imgs_samples)
                # writer.add_graph(generator, enc_imgs)
                # writer.add_graph(discriminator, real_imgs_samples)
                #


        # if epoch % opt.sample_interval == 0 :
        #     sampling(fixed_noise, generator, path_data, epoch, tag)
        #     # do_plot(hist, start_epoch, epoch)

        print("[Epoch Time: ", time.time() - t_epoch, "s]")

    sampling(fixed_noise, generator, path_data, epoch, tag, nrow=16)

    # for scheduler in schedulers: scheduler.step()
    t_final = time.gmtime(time.time() - t_total)
    print("[Total Time: ", t_final.tm_mday - 1, "j:",
          time.strftime("%Hh:%Mm:%Ss", t_final), "]", sep='')

    if do_tensorboard:
        writer.close()
