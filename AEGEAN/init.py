import os
import argparse

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs_path", type=str, default='vanilla',
                        help="Dossier de stockage des résultats pour Tensorboard sous la forme : Experience_names/parameters/")
    parser.add_argument("-e", "--n_epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lrE", type=float, default=0.00015, help="adam: learning rate for E")
    parser.add_argument("--lrD", type=float, default=0.00005, help="adam: learning rate for D")
    parser.add_argument("--lrG", type=float, default=0.00005, help="adam: learning rate for G")
    parser.add_argument("--bn_eps", type=float, default=2e-01, help="batchnorm: espilon for numerical stability")
    parser.add_argument("--bn_momentum", type=float, default=.1, help="batchnorm: momentum for numerical stability")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lrelu", type=float, default=0.000001, help="LeakyReLU : alpha")
    parser.add_argument("--kernel_size", type=int, default=9, help="size of the channel 0")
    parser.add_argument("--stride", type=int, default=2, help="size of the channel 0")
    parser.add_argument("--padding", type=int, default=4, help="size of the channel 0")
    parser.add_argument("--channel0", type=int, default=64, help="size of the channel 0")
    parser.add_argument("--channel1", type=int, default=128, help="size of the channel 1")
    parser.add_argument("--channel2", type=int, default=256, help="size of the channel 2")
    parser.add_argument("--channel3", type=int, default=512, help="size of the channel 3")
    parser.add_argument("--latent_dim", type=int, default=16, help="dimensionality of the latent space")
    parser.add_argument("-i", "--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("-s", "--sample_interval", type=int, default=2, help="interval between image sampling")
    parser.add_argument("--N_samples", type=int, default=49, help="number of samples to generate each time")
    parser.add_argument("--sample_path", type=str, default='images')
    parser.add_argument("-m", "--model_save_interval", type=int, default=5000,
                        help="interval between image sampling. If model_save_interval > n_epochs : no save")
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--datapath', type=str, default='../cropped_clear/cp/')
    parser.add_argument('--load_model', action="store_true",
                        help="Load model present in model_save_path/Last_*.pt, if present.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Afficher des informations complémentaire.")
    parser.add_argument("--GPU", type=int, default=0, help="Identifiant du GPU à utiliser.")
    parser.add_argument("--do_ian_loss", type=bool, default=True, help="Use the loss from Ian Goodfellow.")
    opt = parser.parse_args()

    # Dossier de sauvegarde
    os.makedirs(opt.model_save_path, exist_ok=True)

    if opt.verbose:
        print(opt)
    return opt
