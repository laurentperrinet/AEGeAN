import os
from itertools import product
import time
import torch
import random
import numpy as np
import datetime
# import pathlib
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision
from torchvision.utils import save_image
from PIL import Image
import matplotlib
matplotlib.use('Agg')

# from .SimpsonsDataset import *

# TODO use inception score from https://github.com/Zeleni9/pytorch-wgan/blob/master/utils/inception_score.py
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy

def get_inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """
        Computes the inception score of the generated images imgs
        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

class FolderDataset(Dataset):
    def __init__(self, dir_path, height, width, transform):
        """
        Args:
                dir_path (string): path to dir that contains exclusively png images
                height (int): image height
                width (int): image width
                transform: pytorch transforms for transforms and tensor conversion during training
        """
        files = []
        for ext in ['png', 'PNG', 'jpg', 'JPG']:
            files.extend(glob(os.path.join(dir_path, f'*.{ext}')))
            files.extend(glob(os.path.join(dir_path, f'**/*.{ext}')))
        #self.files = list(pathlib.Path(dir_path).rglob("*.png"))#".[png|jpg]")
        # print(self.files, os.path.join(dir_path, '**/*'))
        # self.labels = np.zeros(len(self.files))
        self.height = height
        self.width = width
        self.transform = transform

        # Chargement des images
        self.imgs = list()
        self.files = list()
        for fname in files:
            if os.path.isfile(fname):
                #img_as_np = np.asarray(Image.open(img).resize((self.height, self.width))).astype('uint8')
                img_as_pil = Image.open(fname).resize((self.height, self.width))

                self.imgs.append(img_as_pil)
                self.files.append(fname)

    def __getitem__(self, index):
        #print("Image load : ",self.files[index])
        #single_image_label = self.labels[index]
        filename = self.files[index]
        img_as_pil = self.imgs[index]

        # Transform image to tensor
        img_as_tensor = self.transform(img_as_pil)

        # Return image and the label
        return (img_as_tensor, filename)

    def __len__(self):
        return len(self.files)

# https://github.com/Kenneth111/pytorch_tutorial_exercises/blob/master/image_transforms.py

from random import random
import numpy as np
from skimage.transform import AffineTransform, warp
from PIL import Image

class ShiftTransform(object):
    def __init__(self, x, y):
        """
        :param x(float): fraction of total width, 0 < x < 1.0
        :param y(float): fraction of total height, 0 < y < 1.0
        """
        super(ShiftTransform, self).__init__()
        self.x = x
        self.y = y

    def __call__(self, img):
        """
        :param img: PIL Image
        :return: PIL Image
        """
        x = int((random() - 0.5) / 0.5 * self.x * img.size[0])
        y = int((random() - 0.5) / 0.5 * self.y * img.size[1])
        tmp_img = np.array(img)
        transform = AffineTransform(translation=(x, y))
        shifted_img = warp(tmp_img, transform, mode="edge", preserve_range=True)
        shifted_img = shifted_img.astype(tmp_img.dtype)
        return Image.fromarray(shifted_img)

    def __repr__(self):
        return self.__class__.__name__ + "(x: {}, y: {})".format(self.x, self.y)


class RotoTransform(object):
    def __init__(self, theta):
        """
        :param theta(float): angle in degrees
        """
        super(RotoTransform, self).__init__()
        self.theta = theta

    def __call__(self, img):
        """
        :param img: PIL Image
        :return: PIL Image
        """
        theta = 2*(np.random.rand() - 0.5) * self.theta
        tmp_img = np.array(img)
        # https://scikit-image.org/docs/dev/api/skimage.transform.html?highlight=affine#skimage.transform.AffineTransform
        transform = AffineTransform(rotation=theta*np.pi/180)
        # https://scikit-image.org/docs/dev/api/skimage.transform.html?highlight=affine#skimage.transform.warp
        shifted_img = warp(tmp_img, transform, mode="edge", preserve_range=True)
        shifted_img = shifted_img.astype(tmp_img.dtype)
        return Image.fromarray(shifted_img)

    def __repr__(self):
        return self.__class__.__name__ + "(theta: {})".format(self.theta)



class Normalize(object):
    def __init__(self, min, max, do_median=True):
        """
        :param max(float): max value
        """
        super(Normalize, self).__init__()
        self.min, self.max = min, max
        self.do_median = do_median

    def __call__(self, img):
        """
        :param img: PIL Image
        :return: PIL Image

        TODO: determine a contrast
        """
        tmp_img = np.array(img).astype(np.float)
        # print(tmp_img.min(), tmp_img.max())
        # if self.do_median:
        #     tmp_img -= np.median(tmp_img)
        # else:
        #     tmp_img -= np.mean(tmp_img)
        tmp_img -= np.min(tmp_img)
        tmp_img = tmp_img / tmp_img.max()
        # tmp_img = tmp_img.astype(tmp_img.dtype)
        # print(tmp_img.min(), tmp_img.max())
        # return Image.fromarray(tmp_img)
        return (self.max-self.min)*tmp_img - self.min

    def __repr__(self):
        return self.__class__.__name__ + "(mean: {}, max: {})".format(self.mean, self.max)


import torch.nn as nn

def weights_init_normal(m, weight_0=0.01, factor=1.0):
    # see https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#weight-initialization
    classname = m.__class__.__name__
    # print('classname', classname)
    if classname.find("Conv") != -1:
        # n = float(m.in_channels * m.kernel_size[0] * m.kernel_size[1])
        # n += float(m.kernel_size[0] * m.kernel_size[1] * m.out_channels)
        # n = n / 2.0
        # m.weight.data.normal_(0, np.sqrt(factor / n))
        # m.bias.data.zero_()
        nn.init.normal_(m.weight.data, 0.0, weight_0*factor)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0.)
    #
    # elif classname.find('Conv2d') != -1:
    #     print('classname', classname)
    #     nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        n = float(m.in_features + m.out_features)
        n = n / 2.0
        m.weight.data.normal_(0, np.sqrt(factor / n))
        m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, weight_0*factor)
        nn.init.constant_(m.bias.data, 0)
        # m.weight.data.fill_(1.0)
        # m.bias.data.zero_()


def load_data(path, img_size, batch_size,
              rand_hflip=False, rand_affine=0.,
              min=0., max=1., mean=0.5, std=1.):
    print("Loading data...")
    t_total = time.time()

    # Sequence of transformations
    transform_tmp = []
    if rand_hflip:
        transform_tmp.append(transforms.RandomHorizontalFlip(p=0.5))
        # transform_tmp.append(ShiftTransform(x=0.05, y=0.05))
    if rand_affine > 0.:
        # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomAffine
        # transform_tmp.append(transforms.RandomAffine(degrees=rand_affine, fillcolor=1))
        transform_tmp.append(RotoTransform(theta=rand_affine))
    # transform_tmp.append(transforms.ColorJitter(brightness=0, contrast=(0.9, 1.0), saturation=0, hue=0))
    transform_tmp.append(Normalize(min, max))
    transform_tmp.append(transforms.ToTensor())
    # transform_tmp.append(transforms.Normalize([mean]*3, [std]*3))

    transform = transforms.Compose(transform_tmp)
    dataset = FolderDataset(path, img_size, img_size, transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("[Loading Time: ", time.strftime("%Mm:%Ss", time.gmtime(time.time() - t_total)),
          "] [Numbers of samples :", len(dataset), " ]\n")

    return dataloader

#
# def save_model(model, optimizer, epoch, path):
#     print("Save model : ", model._name())
#     info = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#     }
#     torch.save(info, path)

#
# def load_model(model, optimizer, path):
#     print("Load model :", model._name())
#     checkpoint = torch.load(path)
#
#     model.load_state_dict(checkpoint['model_state_dict'])
#
#     if optimizer is not None:
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#     return checkpoint['epoch']
#
#
# def load_models(discriminator, optimizer_D, generator, optimizer_G, n_epochs, model_save_path, encoder=None, optimizer_E=None):
#     start_epochD = load_model(discriminator, optimizer_D, model_save_path + "/last_D.pt")
#     start_epochG = load_model(generator, optimizer_G, model_save_path + "/last_G.pt")
#
#     if encoder is not None:
#         start_epochE = load_model(encoder, optimizer_E, model_save_path + "/last_E.pt")
#
#     if start_epochG is not start_epochD:
#         print("Something seems wrong : epochs used to train G and D are different !!")
#         # exit(0)
#     start_epoch = start_epochD
#     # if start_epoch >= n_epochs:
#     #    print("Something seems wrong : you epochs demander inférieur au nombre d'epochs déjà effectuer !!")
#     #    #exit(0)
#
#     return start_epoch + 1  # Last epoch already done


def sampling(noise, generator, path, epoch, tag='', nrow=8):
    """
    Use generator model and noise vector to generate images.
    Save them to path/epoch.png

    """
    generator.eval()
    gen_imgs = generator(noise)
    save_image(gen_imgs.data[:], "%s/%s_%d.png" %
               (path, tag, epoch), normalize=True, nrow=nrow, range=(0, 1))
    generator.train()


def tensorboard_sampling(noise, generator, writer, epoch, nrow=8, image_type='Generated images'):
    """
    Use generator model and noise vector to generate images.
    Save them to tensorboard
    """
    generator.eval()
    gen_imgs = generator(noise)
    grid = torchvision.utils.make_grid(gen_imgs, normalize=True, nrow=nrow, range=(0, 1))
    writer.add_image(image_type, grid, epoch)
    generator.train()


def tensorboard_AE_comparator(imgs, generator, encoder, writer, epoch, nrow=8):
    """
    Use auto-encoder model and original images to generate images.
    Save them to tensorboard

    """
    grid_imgs = torchvision.utils.make_grid(imgs, normalize=True, nrow=nrow, range=(0, 1))
    writer.add_image('Images/original', grid_imgs, epoch)

    generator.eval()
    encoder.eval()
    enc_imgs = encoder(imgs)
    dec_imgs = generator(enc_imgs)
    grid_dec = torchvision.utils.make_grid(dec_imgs, normalize=True, nrow=nrow, range=(0, 1))
    writer.add_image('Images/auto-encoded', grid_dec, epoch)
    generator.train()
    encoder.train()
#
#
# def tensorboard_LSD_comparator(imgs, vectors, generator, writer, epoch, nrow=8):
#     """
#     Use auto-encoder model and noise vector to generate images.
#     Save them to tensorboard
#
#     """
#     writer.add_image('Images/original', grid_imgs, epoch)
#
#     generator.eval()
#     g_v = generator(vectors)
#     grid_imgs = torchvision.utils.make_grid(imgs, normalize=True, nrow=nrow, range=(0, 1))
#     grid_g_v = torchvision.utils.make_grid(g_v, normalize=True, nrow=nrow, range=(0, 1))
#     writer.add_image('Images/generated', grid_g_v, epoch)
#     generator.train()
#

def AE_sampling(imgs, encoder, generator, path, epoch, nrow=8):
    generator.eval()
    enc_imgs = encoder(imgs)
    dec_imgs = generator(enc_imgs)
    save_image(imgs.data[:16], "%s/%d_img.png" %
               (path, epoch), nrow=nrow, normalize=True, range=(0, 1))
    save_image(dec_imgs.data[:16], "%s/%d_dec.png" %
               (path, epoch), nrow=nrow, normalize=True, range=(0, 1))
    generator.train()


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    print()


def generate_animation(path, fps=1):
    import imageio
    images_path = glob(path + '[0-9]*.png')
    images_path = sorted(images_path, key=comp)
    images = []
    for i in images_path:
        # print(i)
        images.append(imageio.imread(i))

    imageio.mimsave(path + 'training.gif', images, fps=fps)

#
# def scan(exp_name, params, permutation=True, gpu_repart=False):
#     """
#     Lance le fichier dcgan.py présent dans le dossier courant avec toutes les combinaisons de paramètres possible.
#     exp_name : Une chaîne de caractère utiliser pour nommer le sous dossier de résultats tensorboard.
#     params : Un dictionnaire où les clefs sont des noms de paramètre (ex : --lrG) et les valeurs sont les différentes
#             valeurs à tester pour ce paramètre.
#     permutation : Si == True alors toute les permutations (sans répétition) possible de params sont tester,
#                   Sinon tout les paramètres sont ziper (tout les paramètres doivent contenir le même nombres d'éléments).
#     gpu_repart (Non fonctionnel) : Si plusieurs GPU sont disponible les commandes seront répartis entre eux.
#     """
#   # Création d'une liste contenant les liste de valeurs à tester
#     val_tab = list()
#     for v in params.values():
#         val_tab.append(v)
#         # print(v)
#     # print(val_tab)
#
#     # Création d'une liste contenant tout les combinaisons de paramètres à tester
#     if permutation:
#         perm = list(product(*val_tab))
#     else:
#         perm = list(zip(*val_tab))
#     # print(perm)
#
#     # Construction du noms de chaque test en fonction des paramètre qui la compose
#     names = list()
#     for values in perm:
#         b = values
#         e = params.keys()
#         l = list(zip(e, b))
#         l_str = [str(ele) for el in l for ele in el]
#         names.append(''.join(l_str).replace('-', ''))
#     # print(names)
#
#     # Construction de toutes les commandes à lancer
#     base = "python3 dcgan.py -r " + exp_name + "/"
#     commandes = list()
#     for j, values in enumerate(perm):
#         com = base + names[j] + "/"
#         for i, a in enumerate(params.keys()):
#             com = com + " " + str(a) + " " + str(values[i])
#         print(com)
#         commandes.append(com)
#     print("Nombre de commande à lancer :", len(commandes))
#
#     # Demande de validation
#     print("Valider ? (Y/N)")
#     reponse = input()
#     #reponse = 'Y'
#     if reponse == 'N':
#         print("Annulation !")
#         exit(0)
#
#     # Appelle successif des script avec différents paramètres
#     log = list()
#     for com in commandes:
#         print("Lancement de : ", com)
#         ret = os.system(com)
#         log.append(ret)
#
#     # Récapitulatif
#     for idx, com in enumerate(commandes):
#         print("Code retour : ", log[idx], "\t| Commandes ", com)
#

if __name__ == "__main__":

    """D_G_z = np.random.normal(0.5,0.5,100)
    D_x = np.random.normal(0.5,0.5,100)

    plot_scores(D_x,D_G_z)

    print("test")"""

    # generate_animation("W7_128_dcgan/gif/")

    # DataLoader test
    loader, dataset = load_data("../cropped_clear/cp/", 200, 6, Fast=True, rand_hflip=True,
                                rand_affine=[(-25, 25), (1.0, 1.0)], return_dataset=True)

    for (imgs, _) in loader:
        show_tensor(imgs[1], 1)
        print("Max ", imgs[1].max())
        print("Min ", imgs[1].min())
        break
        # exit(0)
