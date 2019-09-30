import os
from itertools import product
import time
import torch
import random
import numpy as np
import datetime
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


class FolderDataset(Dataset):
    def __init__(self, dir_path, height, width, transform):
        """
        Args:
                dir_path (string): path to dir that contains exclusively png images
                height (int): image height
                width (int): image width
                transform: pytorch transforms for transforms and tensor conversion during training
        """
        self.files = glob(dir_path + '*')
        self.labels = np.zeros(len(self.files))
        self.height = height
        self.width = width
        self.transform = transform

        # Chargement des images
        self.imgs = list()
        for img in self.files:
            #img_as_np = np.asarray(Image.open(img).resize((self.height, self.width))).astype('uint8')
            img_as_img = Image.open(img).resize((self.height, self.width))

            self.imgs.append(img_as_img)

    def __getitem__(self, index):
        #print("Image load : ",self.files[index])
        single_image_label = self.labels[index]
        img_as_img = self.imgs[index]

        # Transform image to tensor
        img_as_tensor = self.transform(img_as_img)

        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.files)

import torch.nn as nn

def weights_init_normal(m, factor=1.0):
    # see https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#weight-initialization
    classname = m.__class__.__name__
    #print('classname', classname)
    if classname.find("Conv") != -1:
        # n = float(m.in_channels * m.kernel_size[0] * m.kernel_size[1])
        # n += float(m.kernel_size[0] * m.kernel_size[1] * m.out_channels)
        # n = n / 2.0
        # m.weight.data.normal_(0, np.sqrt(factor / n))
        # m.bias.data.zero_()
        nn.init.normal_(m.weight.data, 0.0, 0.02)
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
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        # m.weight.data.fill_(1.0)
        # m.bias.data.zero_()


def load_data(path, img_size, batch_size, Fast=True, FDD=False, rand_hflip=False, rand_affine=None, mean=0., std=1.):
    print("Loading data...")
    t_total = time.time()

    # Sequence of transformations
    transform_tmp = []
    if rand_hflip:
        transform_tmp.append(transforms.RandomHorizontalFlip(p=0.5))
    if rand_affine != None:
        transform_tmp.append(transforms.RandomAffine(degrees=rand_affine))
    # transform_tmp.append(transforms.ColorJitter(brightness=0, contrast=(0.9, 1.0), saturation=0, hue=0))
    transform_tmp.append(transforms.ToTensor())
    # transform_tmp.append(transforms.Normalize([mean]*3, [std]*3))

    transform = transforms.Compose(transform_tmp)
    dataset = FolderDataset(path, img_size, img_size, transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("[Loading Time: ", time.strftime("%Mm:%Ss", time.gmtime(time.time() - t_total)),
          "] [Numbers of samples :", len(dataset), " ]\n")

    return dataloader


def save_model(model, optimizer, epoch, path):
    print("Save model : ", model._name())
    info = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(info, path)


def load_model(model, optimizer, path):
    print("Load model :", model._name())
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']


def load_models(discriminator, optimizer_D, generator, optimizer_G, n_epochs, model_save_path, encoder=None, optimizer_E=None):
    start_epochD = load_model(discriminator, optimizer_D, model_save_path + "/last_D.pt")
    start_epochG = load_model(generator, optimizer_G, model_save_path + "/last_G.pt")

    if encoder is not None:
        start_epochE = load_model(encoder, optimizer_E, model_save_path + "/last_E.pt")

    if start_epochG is not start_epochD:
        print("Something seems wrong : epochs used to train G and D are different !!")
        # exit(0)
    start_epoch = start_epochD
    # if start_epoch >= n_epochs:
    #    print("Something seems wrong : you epochs demander inférieur au nombre d'epochs déjà effectuer !!")
    #    #exit(0)

    return start_epoch + 1  # Last epoch already done


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


def tensorboard_LSD_comparator(imgs, vectors, generator, writer, epoch, nrow=8):
    """
    Use auto-encoder model and noise vector to generate images.
    Save them to tensorboard

    """
    writer.add_image('Images/original', grid_imgs, epoch)

    generator.eval()
    g_v = generator(vectors)
    grid_imgs = torchvision.utils.make_grid(imgs, normalize=True, nrow=nrow, range=(0, 1))
    grid_g_v = torchvision.utils.make_grid(g_v, normalize=True, nrow=nrow, range=(0, 1))
    writer.add_image('Images/generated', grid_g_v, epoch)
    generator.train()


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


def scan(exp_name, params, permutation=True, gpu_repart=False):
    """
    Lance le fichier dcgan.py présent dans le dossier courant avec toutes les combinaisons de paramètres possible.
    exp_name : Une chaîne de caractère utiliser pour nommer le sous dossier de résultats tensorboard.
    params : Un dictionnaire où les clefs sont des noms de paramètre (ex : --lrG) et les valeurs sont les différentes
            valeurs à tester pour ce paramètre.
    permutation : Si == True alors toute les permutations (sans répétition) possible de params sont tester,
                  Sinon tout les paramètres sont ziper (tout les paramètres doivent contenir le même nombres d'éléments).
    gpu_repart (Non fonctionnel) : Si plusieurs GPU sont disponible les commandes seront répartis entre eux.
    """
  # Création d'une liste contenant les liste de valeurs à tester
    val_tab = list()
    for v in params.values():
        val_tab.append(v)
        # print(v)
    # print(val_tab)

    # Création d'une liste contenant tout les combinaisons de paramètres à tester
    if permutation:
        perm = list(product(*val_tab))
    else:
        perm = list(zip(*val_tab))
    # print(perm)

    # Construction du noms de chaque test en fonction des paramètre qui la compose
    names = list()
    for values in perm:
        b = values
        e = params.keys()
        l = list(zip(e, b))
        l_str = [str(ele) for el in l for ele in el]
        names.append(''.join(l_str).replace('-', ''))
    # print(names)

    # Construction de toutes les commandes à lancer
    base = "python3 dcgan.py -r " + exp_name + "/"
    commandes = list()
    for j, values in enumerate(perm):
        com = base + names[j] + "/"
        for i, a in enumerate(params.keys()):
            com = com + " " + str(a) + " " + str(values[i])
        print(com)
        commandes.append(com)
    print("Nombre de commande à lancer :", len(commandes))

    # Demande de validation
    print("Valider ? (Y/N)")
    reponse = input()
    #reponse = 'Y'
    if reponse == 'N':
        print("Annulation !")
        exit(0)

    # Appelle successif des script avec différents paramètres
    log = list()
    for com in commandes:
        print("Lancement de : ", com)
        ret = os.system(com)
        log.append(ret)

    # Récapitulatif
    for idx, com in enumerate(commandes):
        print("Code retour : ", log[idx], "\t| Commandes ", com)


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

        exit(0)
