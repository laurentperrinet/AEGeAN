import os
import argparse

PID, HOST = os.getpid(), os.uname()[1]

if HOST == 'ada':
    DEBUG = 4
elif HOST == 'fortytwo':
    DEBUG = 2
else:
    DEBUG = 1

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_path", type=str, default='vanilla',
                        help="folder to save samples data and statistics")
    parser.add_argument("--n_epochs", type=int, default=32//DEBUG, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--rand_hflip", type=bool, default=True, help="data augmentation: horizontal flip")
    parser.add_argument("--rand_affine", type=float, default=10, help="data augmentation: angle in degrees")
    parser.add_argument("--lrE", type=float, default=0.015*DEBUG, help="adam: learning rate for E")
    parser.add_argument("--lrD", type=float, default=0.00005*DEBUG, help="adam: learning rate for D")
    parser.add_argument("--valid_smooth", type=float, default=1.0, help="Smoothing the results of D on real images")
    parser.add_argument("--D_noise", type=float, default=0.3, help="Add noise to the input images of D")
    parser.add_argument("--lrG", type=float, default=0.0005*DEBUG, help="adam: learning rate for G")
    parser.add_argument("--G_loss", type=str, default='original', help="Use different loss for G.")
    parser.add_argument("--do_whitening", type=bool, default=False, help="Use contrasted images.")
    parser.add_argument("--bn_eps", type=float, default=.5, help="batchnorm: espilon for numerical stability")
    parser.add_argument("--bn_momentum", type=float, default=.1, help="batchnorm: momentum for numerical stability")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lrelu", type=float, default=0.000001, help="LeakyReLU : alpha")
    parser.add_argument("--kernel_size", type=int, default=9, help="size of the channel 0")
    parser.add_argument("--stride", type=int, default=2, help="size of the channel 0")
    parser.add_argument("--padding", type=int, default=4, help="size of the channel 0")
    parser.add_argument("--channel0", type=int, default=64//DEBUG, help="size of the channel 0")
    parser.add_argument("--channel1", type=int, default=128//DEBUG, help="size of the channel 1")
    parser.add_argument("--channel2", type=int, default=128//DEBUG, help="size of the channel 2")
    parser.add_argument("--channel3", type=int, default=128//DEBUG, help="size of the channel 3")
    parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128//DEBUG, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of input image channels")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval in epochs between image sampling")
    parser.add_argument("--N_samples", type=int, default=48, help="number of samples to generate each time")
    parser.add_argument("--model_save_interval", type=int, default=5000,
                        help="interval between image sampling. If model_save_interval > n_epochs : no save")
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--datapath', type=str, default='../cropped_clear/cp/')
    parser.add_argument('--load_model', action="store_true",
                        help="Load model present in model_save_path/Last_*.pt, if present.")
    parser.add_argument("--verbose", type=bool, default=False, help="Displays more verbose output.")
    parser.add_argument("--GPU", type=int, default=0, help="Identifiant du GPU à utiliser.")
    opt = parser.parse_args()

    # Dossier de sauvegarde
    os.makedirs(opt.model_save_path, exist_ok=True)

    if opt.verbose:
        print(opt)
    return opt
