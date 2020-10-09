import os
import numpy as np
import argparse

PID, HOST = os.getpid(), os.uname()[1]
if HOST in ['ada', 'ekla']:
    DEBUG = 4
elif HOST == 'fortytwo':
    DEBUG = 4
    DEBUG = 1
else:
    DEBUG = 1


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, default='vanilla',
                        help="TensorBoard folder to save samples data and statistics")
    parser.add_argument("--n_epochs", type=int, default=2048,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--rand_hflip", type=bool, default=True, help="data augmentation: horizontal flip")
    parser.add_argument("--rand_affine", type=float, default=2., help="data augmentation: angle in degrees")
    parser.add_argument("--init_weight", type=bool, default=True, help="initialize weights to normal")
    parser.add_argument("--gamma", type=float, default=1., help="gamma correction of images")
    parser.add_argument("--lambdaE", type=float, default=.02, help="regularization parameter for E")
    parser.add_argument("--lambdaG", type=float, default=.02, help="regularization parameter for G")
    parser.add_argument("--lrE", type=float, default=0.0003, help="learning rate for E")
    parser.add_argument("--lrD", type=float, default=0.0002, help="learning rate for D")
    parser.add_argument("--lrG", type=float, default=0.0004, help="learning rate for G supervised by D")
    parser.add_argument("--valid_smooth", type=float, default=.94, help="Smoothing the results of D on real images")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout generalization factor in the encoder & generator")
    parser.add_argument("--E_noise", type=float, default=0.01, help="Add noise to the input images to G_E_x")
    parser.add_argument("--D_noise", type=float, default=0., help="Add noise to the input images to D_x")
    parser.add_argument("--G_noise", type=float, default=0.001, help="Add noise to the input images to D_G_z")
    parser.add_argument("--GAN_loss", type=str, default='original', help="Use different optimizers.")
    parser.add_argument("--optimizer", type=str, default='sgd', help="Use different losses.")
    parser.add_argument("--do_SSIM", type=bool, default=True, help="Use contrasted images for the cost of E.")
    parser.add_argument("--do_bias", type=bool, default=True, help="Should we use biases in convolutions?")
    parser.add_argument("--lrelu", type=float, default=0., help="LeakyReLU : alpha - zero for a standard ReLU")
    parser.add_argument("--do_slerp", type=bool, default=True, help="Draw random vectors between E(x1) and  E(x2).")
    parser.add_argument("--do_joint", type=bool, default=True, help="Do a joint learning of E and G, dude.")
    parser.add_argument("--do_insight", type=bool, default=False, help="D looks at G_E_x instead of x.")
    parser.add_argument("--do_transpose", type=bool, default=False, help="use of Conv2Dtranspose.")
    parser.add_argument("--bn_eps", type=float, default=0.01, help="norm: espilon for numerical stability")
    parser.add_argument("--bn_momentum", type=float, default=.1,
                        help="batchnorm: momentum for numerical stability")
    parser.add_argument("--beta1", type=float, default=0.985,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.99,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--resblocks", type=int, default=4, help="number of ResNet  block")
    parser.add_argument("--channel0_bg", type=int, default=2, help="size of the background mask channel")
    parser.add_argument("--channel0", type=int, default=8, help="size of channel 0")
    parser.add_argument("--channel1", type=int, default=16, help="size of channel 1")
    parser.add_argument("--channel2", type=int, default=32, help="size of channel 2")
    parser.add_argument("--channel3", type=int, default=32, help="size of channel 3")
    parser.add_argument("--channel4", type=int, default=512, help="size of (linear) layer 4")
    parser.add_argument("--latent_dim", type=int, default=42, help="dimensionality of the latent space")
    parser.add_argument("--kernel_size", type=int, default=5, help="size of the kernels")
    parser.add_argument("--stride", type=int, default=2, help="stride")
    parser.add_argument("--padding", type=int, default=2, help="padding")
    parser.add_argument("--padding_mode", type=str, default='reflect', help="Handling values outside the range.")
    parser.add_argument("--img_size", type=int, default=128//DEBUG, help="size of each image dimension")
    parser.add_argument("--window_size", type=int, default=16, help="size of window_size for SSIM")
    parser.add_argument("--channels", type=int, default=3, help="number of input image channels")
    parser.add_argument("--sample_interval", type=int, default=128, help="interval in epochs between image sampling")
    parser.add_argument("--N_samples", type=int, default=48, help="number of images each sampling")
    # parser.add_argument("--model_save_interval", type=int, default=5000,
    #                     help="interval between image sampling. If model_save_interval > n_epochs : no save")
    # parser.add_argument('--model_save_path', type=str, default='models')
    # parser.add_argument('--datapath', type=str, default='../database/Simpsons-Face_clear/cp/')
    parser.add_argument('--datapath', type=str, default='../database/CFD Version 2.0.3/CFD 2.0.3 Images')
    # parser.add_argument('--load_model', action="store_true",
    #                     help="Load model present in model_save_path/Last_*.pt, if present.")
    parser.add_argument("--verbose", type=bool, default=False if DEBUG < 4 else True,
                                     help="Displays more verbose output.")
    opt = parser.parse_args()

    # Dossier de sauvegarde
    # os.makedirs(opt.model_save_path, exist_ok=True)

    if opt.verbose:
        print(opt)
    return opt
