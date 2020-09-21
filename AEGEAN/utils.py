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
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')

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
                img_as_pil = Image.open(fname)
                # HACK for CFD images
                # if list(img_as_pil.getdata())[0]  == (255, 255, 255): # rgb_im.getpixel((1, 1))
                #     ImageDraw.floodfill(img_as_pil, xy=(0, 0), value=(127, 127, 127), thresh=10)
                #     ImageDraw.floodfill(img_as_pil, xy=(0, -1), value=(127, 127, 127), thresh=10)
                img_as_pil = img_as_pil.resize((self.height, self.width), resample=Image.BILINEAR)
                # TODO: use LAB color space for CFD / CMYK for simpsons
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
    def __init__(self, min, mean, max):
        """
        :param max(float): max value
        """
        super(Normalize, self).__init__()
        self.min, self.mean, self.max = min, mean, max
        # self.do_median = do_median

    def __call__(self, img):
        """
        :param img: PIL Image
        :return: PIL Image

        """
        tmp_img = np.array(img).astype(np.float)
        #print('min-mean-max', tmp_img.min(), tmp_img.mean(), tmp_img.max())
        if False:
            tmp_img -= np.mean(tmp_img)
            tmp_img /= np.abs(tmp_img).max()
            tmp_img *= self.max - self.mean
            tmp_img += self.mean
        else:
            tmp_img -= np.min(tmp_img)
            tmp_img = tmp_img / tmp_img.max()
            # tmp_img = tmp_img.astype(tmp_img.dtype)
            # print('0-1', tmp_img.min(), tmp_img.max())
            # return Image.fromarray(tmp_img)
            tmp_img = (self.max-self.min)*tmp_img + self.min
        #print('min-mean-max', tmp_img.min(), tmp_img.mean(), tmp_img.max())
        return tmp_img

    def __repr__(self):
        return self.__class__.__name__ + "(mean: {}, max: {})".format(self.mean, self.max)


import torch.nn as nn

def weights_init_normal(m, weight_0=0.01, factor=1.0):
    # see https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#weight-initialization
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, weight_0*factor)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0.)

    elif classname.find("Linear") != -1:
        n = float(m.in_features + m.out_features)
        n = n / 2.0
        m.weight.data.normal_(0, np.sqrt(factor / n))
        m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, weight_0*factor)
        nn.init.constant_(m.bias.data, 0)


def load_data(path, img_size, batch_size,
              rand_hflip=False, rand_affine=0.,
              min=0., max=1., mean=.5, std=1.):
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
    transform_tmp.append(Normalize(min, mean, max))
    transform_tmp.append(transforms.ToTensor())
    # transform_tmp.append(transforms.Normalize([mean]*3, [std]*3))

    transform = transforms.Compose(transform_tmp)

    file_path = f"/tmp/AEGEAN_dataset_{path.replace('/', '_')}_{img_size}.pt"
    print(file_path)
    try :
        dataset = torch.load(file_path)
        print("Loading dataset")
    except :

        dataset = FolderDataset(path, img_size, img_size, transform)
        torch.save(dataset, file_path)

    use_cuda = True if torch.cuda.is_available() else False
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    print("[Loading Time: ", time.strftime("%Mm:%Ss", time.gmtime(time.time() - t_total)),
          "] [Numbers of samples :", len(dataset), " ]\n")

    return dataloader

def sampling(noise, generator, path, epoch, tag='', nrow=16):
    """
    Use generator model and noise vector to generate images.
    Save them to path/epoch.png

    """
    generator.eval()
    gen_imgs = generator(noise)
    save_image(gen_imgs.data[:], f"{path}/{tag}_{epoch:04d}.png", normalize=True, nrow=nrow, range=(0, 1))
    generator.train()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    print()



def init_hist(nb_epochs, nb_batch):
    """
    Initialise et retourne un dictionnaire qui servira à sauvegarder les données que l'on voudrais afficher par la suite.
    """

    hist = {}

    # Container for ploting (en minuscule pour les batchs et en majuscule pour les epochs)
    # Losses G et D
    hist["G_losses"] = np.zeros(nb_epochs)
    hist["D_losses"] = np.zeros(nb_epochs)
    hist["g_losses"] = np.zeros(nb_batch)
    hist["d_losses"] = np.zeros(nb_batch)
    hist["E_losses"] = np.zeros(nb_epochs)
    hist["e_losses"] = np.zeros(nb_batch)

    # Moyenne des réponse D(x) et D(G(z)) moyenner par epochs
    hist["D_x_mean"] = np.zeros(nb_epochs)
    hist["D_G_z_mean"] = np.zeros(nb_epochs)
    hist["d_x_mean"] = np.zeros(nb_batch)
    # hist["d_fake_mean"] = np.zeros(nb_batch)
    hist["d_g_z_mean"] = np.zeros(nb_batch)

    return hist


def save_hist_batch(hist, idx_batch, idx_epoch, g_loss, d_loss, e_loss, d_x, d_g_z): #, d_fake
    """
    Sauvegarde les données du batch dans l'historique après traitement
    """

    d_x = d_x.detach().cpu().numpy()
    # d_fake = d_fake.detach().cpu().numpy()
    d_g_z = d_g_z.detach().cpu().numpy()
    g_loss = g_loss.item()
    d_loss = d_loss.item()
    e_loss = e_loss.item()

    hist["g_losses"][idx_batch] = g_loss
    hist["d_losses"][idx_batch] = d_loss
    hist["e_losses"][idx_batch] = e_loss


    hist["d_x_mean"][idx_batch] = d_x.mean()
    # hist["d_fake_mean"][idx_batch] = d_fake.mean()
    hist["d_g_z_mean"][idx_batch] = d_g_z.mean()


def generate_animation(path, fps=1):
    import imageio
    images_path = glob(path + '[0-9]*.png')
    images_path = sorted(images_path, key=comp)
    images = []
    for i in images_path:
        # print(i)
        images.append(imageio.imread(i))

    imageio.mimsave(path + 'training.gif', images, fps=fps)

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
